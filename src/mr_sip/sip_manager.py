#!/usr/bin/env python3
"""
SIP Session Manager for MindRoot

Manages SIP sessions and their association with MindRoot conversation contexts.
Handles audio routing between SIP calls and MindRoot's TTS/STT systems.
"""

import asyncio
import threading
import logging
from typing import Dict, Optional, Any
from datetime import datetime
import traceback

logger = logging.getLogger(__name__)

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
        self.audio_queue = asyncio.Queue(maxsize=35)  # Increased to ~700ms headroom for smoother pacing
        self._audio_sender_task = None
        self._audio_sent_count = 0
        self._audio_queued_count = 0
        
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
                logger.debug(f"S2S_DEBUG: In while loop, waiting for audio...")
                try:
                    item = await asyncio.wait_for(self.audio_queue.get(), timeout=30.0)
                    if item is None:  # Sentinel to stop
                        break
                    
                    # Unpack audio chunk and timestamp
                    audio_chunk, timestamp = item if isinstance(item, tuple) else (item, None)
                    
                    await self._send_audio_to_sip(audio_chunk, timestamp)
                    self._audio_sent_count += 1
                    if self._audio_sent_count % 10 == 0:
                        logger.info(f"S2S_DEBUG: Sent {self._audio_sent_count} audio chunks to SIP for session {self.log_id}")
                except asyncio.TimeoutError:
                    continue
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
            
    async def _send_audio_to_sip(self, audio_chunk: bytes, timestamp=None):
        """Send audio chunk to the SIP call via JACK."""
        logger.debug(f"S2S_DEBUG: _send_audio_to_sip called with {len(audio_chunk)} bytes")
        if self.baresip_bot and hasattr(self.baresip_bot, 'send_tts_audio'):
            try:
                await self.baresip_bot.send_tts_audio(audio_chunk)
            except Exception as e:
                logger.error(f"Failed to send audio to SIP: {e}")
        else:
            logger.warning(f"No audio output method available for session {self.log_id}")
            
    async def send_audio(self, audio_chunk: bytes, timestamp=None):
        """Queue audio chunk for sending to SIP call"""
        if self.is_active:
            self._audio_queued_count += 1
            try:
                # Phase 1 optimization: Non-blocking put with timeout to prevent queue buildup
                await asyncio.wait_for(
                    self.audio_queue.put((audio_chunk, timestamp)),
                    timeout=0.1
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
            
            # Clear bot's queue if available
            if self.baresip_bot and hasattr(self.baresip_bot, 'clear_audio_queue'):
                self.baresip_bot.clear_audio_queue()
        except Exception as e:
            logger.error(f"Error clearing audio queue: {e}")
                
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
        if self.baresip_bot and hasattr(self.baresip_bot, 'hang'):
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
                await self.end_session(log_id)
                
            session = SIPSession(log_id, destination, baresip_bot)
            self.sessions[log_id] = session
            logger.info(f"Created SIP session {log_id} for destination {destination}")
            return session
            
    async def get_session(self, log_id: str) -> Optional[SIPSession]:
        """Get an existing SIP session"""
        async with self._lock:
            return self.sessions.get(log_id)
            
    async def end_session(self, log_id: str) -> bool:
        """End a SIP session"""
        async with self._lock:
            session = self.sessions.get(log_id)
            if session:
                await session.end_session()
                del self.sessions[log_id]
                logger.info(f"Ended SIP session {log_id}")
                return True
            return False
            
    async def get_active_sessions(self) -> Dict[str, SIPSession]:
        """Get all active sessions"""
        async with self._lock:
            return {log_id: session for log_id, session in self.sessions.items() if session.is_active}
            
    async def cleanup_all_sessions(self):
        """Cleanup all sessions (called on shutdown)"""
        async with self._lock:
            for log_id in list(self.sessions.keys()):
                await self.end_session(log_id)
            logger.info("All SIP sessions cleaned up")

# Global session manager instance
_session_manager = None

def get_session_manager() -> SIPSessionManager:
    """Get or create the global SIP session manager"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SIPSessionManager()
    return _session_manager
