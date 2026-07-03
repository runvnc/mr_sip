#!/usr/bin/env python3
"""
SIP Session Manager for MindRoot

Manages SIP sessions and their association with MindRoot conversation contexts.
Handles audio routing between SIP calls and MindRoot's TTS/STT systems.
"""

import asyncio
import threading
import logging
import time
from typing import Dict, Optional, Any
from datetime import datetime
import traceback

logger = logging.getLogger(__name__)

# End-to-end latency log (shared across mr_sip + PySIP)
E2E_LATENCY_LOG = '/tmp/sip_e2e_latency.log'


def _e2e_log(event: str, utterance_num: int = 0, **kwargs):
    """Log an end-to-end latency event with perf_counter timestamp."""
    from datetime import datetime as _dt
    now = _dt.now()
    ts = now.strftime('%Y-%m-%d %H:%M:%S') + f'.{now.microsecond // 1000:03d}'
    pc = time.perf_counter()
    extra = ' '.join(f'{k}={v}' for k, v in kwargs.items())
    line = f'[{ts}] [E2E] {event} perf_counter={pc:.6f} utterance={utterance_num} {extra}'
    try:
        with open(E2E_LATENCY_LOG, 'a') as f:
            f.write(line + '\n')
            f.flush()
    except Exception:
        pass
    logger.info(f'[E2E] {event} utterance={utterance_num} {extra}')


class SIPSession:
    """
    Represents an active SIP call session linked to a MindRoot conversation.
    """
    
    def __init__(self, log_id: str, destination: str, baresip_bot=None):
        self.log_id = log_id
        self.destination = destination
        self.baresip_bot = baresip_bot
        self.created_at = datetime.now()
        self.is_active = False
        self.halt_audio_out = False
        self._halt_audio_set_time = None  # Track when halt was set for debugging
        
        # Profiling
        self._first_chunk_queued_time: Optional[float] = None
        self._first_chunk_sent_time: Optional[float] = None
        self._e2e_first_chunk_queued_logged: bool = False
        self._e2e_first_chunk_dequeued_logged: bool = False
        self._e2e_current_utterance_num: int = 0
        self.audio_queue = asyncio.Queue(maxsize=35)  # Increased to ~700ms headroom for smoother pacing
        self._audio_sender_task = None
        self._audio_sent_count = 0
        self._audio_queued_count = 0
        # Bytes of u-law actually dequeued+sent for the CURRENT outbound response.
        # Reset at each start_audio_response. Because TTS plugins pace audio to
        # real time before it reaches us, this is a good proxy for how much of
        # the response the caller actually heard before a barge-in. u-law @ 8kHz
        # is 1 byte/sample, so played_seconds = _response_bytes_sent / 8000.
        self._response_bytes_sent = 0
        # Explicit outbound response lifecycle markers are queued through the
        # same audio_queue as audio chunks so start/chunk/end ordering is exact.
        self._audio_response_active = False
        
    async def start_audio_sender(self):
        """Start the audio sender task for TTS output"""
        logger.info(f"S2S_DEBUG: start_audio_sender called for session {self.log_id}")
        logger.info(f"S2S_DEBUG: Current task status: {self._audio_sender_task}")
        if self._audio_sender_task is None:
            self._audio_sender_task = asyncio.create_task(self._audio_sender_loop())
            logger.info(f"S2S_DEBUG: Created new audio sender task: {self._audio_sender_task}")
            
    async def stop_audio_sender(self):
        """Stop the audio sender task"""
        logger.info(f"S2S_DEBUG: Stopping audio sender for session {self.log_id}")
        trace = traceback.format_stack()
        logger.debug(f"S2S_DEBUG: stop_audio_sender called for session {self.log_id}\n{''.join(trace)}")
        if self._audio_sender_task:
            self._audio_sender_task.cancel()
            try:
                await self._audio_sender_task
            except asyncio.CancelledError:
                pass
            self._audio_sender_task = None
            
    async def _audio_sender_loop(self):
        """Background task that sends audio chunks to the SIP call"""
        logger.info(f"S2S_DEBUG: Audio sender loop started for session {self.log_id}")
        logger.info(f"S2S_DEBUG: Session is_active={self.is_active}")
        logger.info(f"S2S_DEBUG: About to enter while loop")
        try:
            while self.is_active:
                if self._bot_call_has_ended():
                    logger.info(
                        "SIP audio sender exiting because bot/PySIP call has ended: "
                        "session=%s bot_ended=%s bot_ending=%s call_state=%s dialogue_state=%s",
                        self.log_id,
                        getattr(self.baresip_bot, '_ended', None),
                        getattr(self.baresip_bot, '_ending', None),
                        self._bot_call_state(),
                        self._bot_dialogue_state(),
                    )
                    self.is_active = False
                    break
                logger.debug(f"S2S_DEBUG: In while loop, waiting for audio...")
                try:
                    item = await asyncio.wait_for(self.audio_queue.get(), timeout=30.0)
                    if self._bot_call_has_ended():
                        logger.info(
                            "SIP audio sender exiting after queue wake because bot/PySIP call has ended: "
                            "session=%s bot_ended=%s bot_ending=%s call_state=%s dialogue_state=%s",
                            self.log_id,
                            getattr(self.baresip_bot, '_ended', None),
                            getattr(self.baresip_bot, '_ending', None),
                            self._bot_call_state(),
                            self._bot_dialogue_state(),
                        )
                        self.is_active = False
                        break
                    if item is None:  # Sentinel to stop
                        break

                    if isinstance(item, dict):
                        command = item.get('command')
                        if command == 'start_audio_response':
                            if self.baresip_bot and hasattr(self.baresip_bot, 'start_tts_response'):
                                await self.baresip_bot.start_tts_response()
                            if hasattr(self.baresip_bot, '_e2e_current_utterance_num'):
                                self._e2e_current_utterance_num = self.baresip_bot._e2e_current_utterance_num
                            # Reset per-response e2e tracking
                            self._e2e_first_chunk_queued_logged = False
                            self._e2e_first_chunk_dequeued_logged = False
                            self._first_chunk_queued_time = None
                            self._first_chunk_sent_time = None
                            self._audio_response_active = True
                            self._response_bytes_sent = 0
                            continue
                        elif command == 'end_audio_response':
                            if self.baresip_bot and hasattr(self.baresip_bot, 'end_tts_response'):
                                await self.baresip_bot.end_tts_response()
                            self._audio_response_active = False
                            continue
                        else:
                            logger.warning(f"Unknown audio queue command for session {self.log_id}: {command}")
                            continue
                    
                    send_start = time.perf_counter()
                    
                    if self._first_chunk_sent_time is None:
                        self._first_chunk_sent_time = send_start
                        if self._first_chunk_queued_time:
                            if not self._e2e_first_chunk_dequeued_logged:
                                _e2e_log('FIRST_CHUNK_DEQUEUED', utterance_num=self._e2e_current_utterance_num, since_queued_ms=f'{(send_start - self._first_chunk_queued_time)*1000:.1f}')
                                self._e2e_first_chunk_dequeued_logged = True
                            queue_latency = (send_start - self._first_chunk_queued_time) * 1000
                            logger.info(f"SIP_SEND: First chunk dequeued, queue_latency={queue_latency:.1f}ms")
                    
                    logger.info(
                        f"SIP_SEND: Sending chunk #{self._audio_sent_count + 1}, queue_size={self.audio_queue.qsize()}"
                    )
                    # Unpack audio chunk and timestamp
                    audio_chunk, timestamp = item if isinstance(item, tuple) else (item, None)
                    
                    await self._send_audio_to_sip(audio_chunk, timestamp)
                    self._audio_sent_count += 1
                    try:
                        self._response_bytes_sent += len(audio_chunk)
                    except Exception:
                        pass
                    if self._audio_sent_count % 10 == 0:
                        logger.info(f"S2S_DEBUG: Sent {self._audio_sent_count} audio chunks to SIP for session {self.log_id}")
                    
                    send_end = time.perf_counter()
                    logger.debug(f"SIP_SEND: _send_audio_to_sip took {(send_end - send_start)*1000:.1f}ms")
                except asyncio.TimeoutError:
                    if self._bot_call_has_ended():
                        logger.info(
                            "SIP audio sender timeout wake found ended bot/PySIP call; exiting session=%s",
                            self.log_id,
                        )
                        self.is_active = False
                        break
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(f"Error in audio sender loop: {e}")
                    break
        except asyncio.CancelledError:
            trace = traceback.format_exc()
            logger.info(f"Audio sender cancelled for session {self.log_id}\n{trace}")
        finally:
            logger.info(f"S2S_DEBUG: Audio sender loop exiting for session {self.log_id}")
            
    def _bot_call_state(self):
        try:
            call = getattr(self.baresip_bot, 'call', None) if self.baresip_bot else None
            return getattr(call, 'call_state', None)
        except Exception:
            return None

    def _bot_dialogue_state(self):
        try:
            call = getattr(self.baresip_bot, 'call', None) if self.baresip_bot else None
            dialogue = getattr(call, 'dialogue', None) if call else None
            return getattr(dialogue, 'state', None)
        except Exception:
            return None

    def _bot_call_has_ended(self) -> bool:
        """Return True if the underlying SIP bot/call is already over.

        This is a defensive backstop for incoming-call cleanup.  If the call-ended
        callback is interrupted or an external hangup path leaves the session
        marked active, the audio sender should not keep logging forever.
        """
        bot = self.baresip_bot
        if not bot:
            return False
        if getattr(bot, '_ended', False):
            return True
        call_state = self._bot_call_state()
        dialogue_state = self._bot_dialogue_state()
        if str(call_state) in ('CallState.ENDED', 'CallState.FAILED', 'CallState.BUSY'):
            return True
        if str(dialogue_state).endswith('TERMINATED'):
            return True
        if getattr(bot, 'is_active', True) is False and getattr(bot, 'call_established', False) is False:
            return True
        return False

    async def _send_audio_to_sip(self, audio_chunk: bytes, timestamp=None):
        """Send audio chunk to the SIP call.

        In S2S mode, `timestamp` is the target playback start time for this
        chunk, generated by the OpenAI AudioPacer. We forward it to the
        underlying bot so PySIP can schedule frames precisely.
        """
        logger.debug(
            f"S2S_DEBUG: _send_audio_to_sip called with {len(audio_chunk)} bytes, "
            f"timestamp={timestamp}"
        )
        if self.baresip_bot and hasattr(self.baresip_bot, 'send_tts_audio'):
            try:
                await self.baresip_bot.send_tts_audio(audio_chunk, timestamp=timestamp)
            except Exception as e:
                logger.error(f"Failed to send audio to SIP: {e}")
        else:
            logger.warning(f"No audio output method available for session {self.log_id}")
            
    async def start_audio_response(self):
        """Start an explicit outbound audio response.

        Calls start_tts_response() directly on the bot instead of queueing
        through audio_queue. This eliminates ~25-30ms of asyncio scheduling
        delay that would occur waiting for the audio_sender_loop to pick up
        the queued command. Ordering is preserved because sip_start_audio_response
        is always called BEFORE any audio chunks are queued.
        """
        if not self.is_active:
            raise RuntimeError("Failed to start audio response: SIP session is not active")
        # Call directly instead of queueing - eliminates asyncio scheduling delay
        if self.baresip_bot and hasattr(self.baresip_bot, 'start_tts_response'):
            await self.baresip_bot.start_tts_response()
        if hasattr(self.baresip_bot, '_e2e_current_utterance_num'):
            self._e2e_current_utterance_num = self.baresip_bot._e2e_current_utterance_num
        # Reset per-response e2e tracking
        self._e2e_first_chunk_queued_logged = False
        self._e2e_first_chunk_dequeued_logged = False
        self._first_chunk_queued_time = None
        self._first_chunk_sent_time = None
        self._audio_response_active = True
        self._response_bytes_sent = 0

    async def end_audio_response(self):
        """Queue an explicit outbound audio response end marker.

        Keep this ordered through audio_queue so the end sentinel reaches PySIP
        only after all earlier queued chunks for this response have been sent to
        the bot.
        """
        if not self.is_active:
            return
        await self.audio_queue.put({'command': 'end_audio_response'})
        self._audio_response_active = False

    async def send_audio(self, audio_chunk: bytes, timestamp=None):
        """Queue audio chunk for sending to SIP call"""
        if self.is_active:
            queue_time = time.perf_counter()
            self._audio_queued_count += 1
            
            if self._first_chunk_queued_time is None:
                self._first_chunk_queued_time = queue_time
                if not self._e2e_first_chunk_queued_logged:
                    _e2e_log('FIRST_CHUNK_QUEUED', utterance_num=self._e2e_current_utterance_num, chunk_len=len(audio_chunk))
                    self._e2e_first_chunk_queued_logged = True
                logger.info(f"SIP_QUEUE: First chunk queued at {queue_time:.3f}")
            
            try:
                # Phase 1 optimization: Non-blocking put with timeout to prevent queue buildup
                queue_size_before = self.audio_queue.qsize()
                await asyncio.wait_for(
                    self.audio_queue.put((audio_chunk, timestamp)),
                    timeout=0.1
                )
                logger.info(
                    f"SIP_QUEUE: chunk #{self._audio_queued_count}, {len(audio_chunk)} bytes, "
                    f"queue_size={queue_size_before}->{self.audio_queue.qsize()}"
                )
                #logger.debug(f"S2S_DEBUG: Queued audio chunk #{self._audio_queued_count}, queue size: {self.audio_queue.qsize()}")
                if self._audio_queued_count % 10 == 0:
                    logger.info(f"S2S_DEBUG: Total queued: {self._audio_queued_count}, sent: {self._audio_sent_count}, queue size: {self.audio_queue.qsize()}")
            except asyncio.TimeoutError:
                # Queue full - drop this chunk to prevent latency accumulation
                logger.warning(f"Audio queue full for session {self.log_id}, dropping chunk")
                return
            except Exception as e:
                logger.error(f"Failed to queue audio chunk: {e}")
        else:
            raise RuntimeError("Failed to queue audio chunk: SIP session is not active")
                
    def clear_audio_queue(self):
        """Clear all queued audio (for interruption)."""
        try:
            # Clear session queue
            cleared_count = 0
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                    cleared_count += 1
                except:
                    break
            logger.info(f"Cleared {cleared_count} chunks from session audio queue")
            self._audio_response_active = False
            
            # Clear bot's queue if available
            if self.baresip_bot and hasattr(self.baresip_bot, 'clear_audio_queue'):
                self.baresip_bot.clear_audio_queue()
        except Exception as e:
            logger.error(f"Error clearing audio queue: {e}")
    
    def halt_audio(self):
        """Halt audio output (for interruption).
        
        This prevents new audio chunks from being sent while allowing
        the call to continue receiving audio.
        """
        import time
        self.halt_audio_out = True
        self._halt_audio_set_time = time.time()
        logger.info(f"Audio output HALTED for session {self.log_id}")
        self.clear_audio_queue()
    
    def resume_audio(self):
        """Resume audio output (when new response starts)."""
        if self.halt_audio_out:
            import time
            halt_duration = time.time() - self._halt_audio_set_time if self._halt_audio_set_time else 0
            logger.info(f"Audio output RESUMED for session {self.log_id} (was halted for {halt_duration:.2f}s)")
        self.halt_audio_out = False
                
    def played_seconds(self) -> float:
        """Approximate seconds of the current/last outbound response actually
        sent to the call (u-law 8kHz, 1 byte/sample). Reset at each
        start_audio_response. Used to estimate how much of an interrupted
        response the caller actually heard before a barge-in.
        """
        try:
            return self._response_bytes_sent / 8000.0
        except Exception:
            return 0.0

    async def end_session(self):
        """End the SIP session and cleanup resources"""
        logger.info(f"Ending SIP session {self.log_id}")

        self.is_active = False
        
        # Signal audio sender to stop
        try:
            await self.audio_queue.put(None)
        except:
            pass
            
        await self.stop_audio_sender()
        
        # Hangup the call if still active
        if (
            self.baresip_bot
            and hasattr(self.baresip_bot, 'hang')
            and not getattr(self.baresip_bot, '_ended', False)
            and not getattr(self.baresip_bot, '_ending', False)
        ):
            try:
                self.baresip_bot.hang()
            except Exception as e:
                logger.error(f"Error hanging up call: {e}")

class SIPSessionManager:
    """
    Manages multiple SIP sessions and their association with MindRoot contexts.
    """
    
    def __init__(self):
        self.sessions: Dict[str, SIPSession] = {}
        self._lock = asyncio.Lock()
        
    async def create_session(self, log_id: str, destination: str, baresip_bot=None) -> SIPSession:
        """Create a new SIP session"""
        async with self._lock:
            if log_id in self.sessions:
                logger.warning(f"Session {log_id} already exists, ending previous session")
                # MUST NOT call self.end_session() here: it re-acquires self._lock
                # (asyncio.Lock is NOT reentrant) and deadlocks, holding the global
                # lock forever and hanging every other get_session/create_session/
                # end_session in the process. Use the lock-free helper.
                await self._end_session_locked(log_id)

            session = SIPSession(log_id, destination, baresip_bot)
            self.sessions[log_id] = session
            logger.info(f"Created SIP session {log_id} for destination {destination}")
            return session
            
    async def get_session(self, log_id: str) -> Optional[SIPSession]:
        """Get an existing SIP session"""
        async with self._lock:
            return self.sessions.get(log_id)
            
    async def _end_session_locked(self, log_id: str) -> bool:
        """End a session. Caller MUST already hold self._lock."""
        session = self.sessions.get(log_id)
        if session:
            await session.end_session()
            del self.sessions[log_id]
            logger.info(f"Ended SIP session {log_id}")
            return True
        return False

    async def end_session(self, log_id: str) -> bool:
        """End a SIP session"""
        async with self._lock:
            return await self._end_session_locked(log_id)
            
    async def get_active_sessions(self) -> Dict[str, SIPSession]:
        """Get all active sessions"""
        async with self._lock:
            return {log_id: session for log_id, session in self.sessions.items() if session.is_active}
            
    async def cleanup_all_sessions(self):
        """Cleanup all sessions (called on shutdown)"""
        async with self._lock:
            for log_id in list(self.sessions.keys()):
                await self._end_session_locked(log_id)
            logger.info("All SIP sessions cleaned up")

# Global session manager instance
_session_manager = None

def get_session_manager() -> SIPSessionManager:
    """Get or create the global SIP session manager"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SIPSessionManager()
    return _session_manager
