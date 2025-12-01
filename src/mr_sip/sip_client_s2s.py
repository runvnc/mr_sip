#!/usr/bin/env python3
"""
SIP Client for Speech-to-Speech Mode using PySIP - OPTIMIZED FOR RTP

Key optimizations:
- Non-blocking audio queue operations
- Critical logging for dropped frames
- Metrics tracking for monitoring
- No blocking operations in audio path
"""

import asyncio
import logging
import queue
import traceback
import time
import audioop
from datetime import datetime
from typing import Optional
from PySIP.sip_call import SipCall
from PySIP.filters import CallState
from lib.providers.services import service_manager
from .call_recorder import CallRecorder, S2SBufferedRecorder

logger = logging.getLogger(__name__)

class AudioStreamAdapter:
    """Adapter to feed audio to PySIP's RTP session.
    
    PySIP expects an object with an input_q attribute (queue.Queue)
    that it reads audio frames from.
    """
    def __init__(self):
        # Queue for low latency - max 1000 frames = 20 seconds buffering
        self.input_q = queue.Queue(maxsize=1000)
        self.stream_id = "tts_output"
        self._done = False
        self.pre_encoded = True  # Flag to indicate audio is already ulaw encoded
    
    def stream_done(self):
        """Mark stream as done."""
        self._done = True

class MindRootSIPBotS2S:
    """SIP phone bot for Speech-to-Speech mode using PySIP - OPTIMIZED.
    
    OUTBOUND CALLS ONLY - Does not handle incoming calls.
    
    Handles bidirectional audio:
    - Input: Phone audio -> OpenAI (via on_frame_received callback)
    - Output: OpenAI audio -> Phone (via send_tts_audio method)
    
    OPTIMIZATION NOTES:
    - All audio operations are non-blocking
    - Dropped frames are logged as CRITICAL
    - Metrics tracked for monitoring
    """
    
    def __init__(self, user: str, password: str, gateway: str, audio_dir: str = ".", 
                 context=None, enable_recording: bool = False, recording_dir: str = "recordings", 
                 record_separate: bool = False,
                 # NEW: Optional queues for process isolation
                 audio_in_queue=None,
                 audio_out_queue=None,
                 status_queue=None):
        """
        Args:
            user: SIP username
            password: SIP password
            gateway: SIP gateway (format: "host:port")
            audio_dir: Unused, kept for compatibility
            context: MindRoot ChatContext
            enable_recording: Enable call recording
            recording_dir: Directory to save recordings
            record_separate: If True, save separate incoming/outgoing files in addition to combined
            audio_in_queue: Optional multiprocessing.Queue for sending audio to main process
            audio_out_queue: Optional multiprocessing.Queue for receiving audio from main process
            status_queue: Optional multiprocessing.Queue for sending status updates (Process Isolation)
        """
        self.sip_username = user
        self.sip_password = password
        
        # Parse gateway - add default port 5060 if not specified
        if ':' in gateway:
            self.sip_server = gateway
        else:
            self.sip_server = f"{gateway}:5060"
            logger.info(f"No port specified in gateway, using default: {self.sip_server}")
        
        self.context = context
        
        # Call state tracking
        self.call: Optional[SipCall] = None
        self.is_active = False
        self.call_established = False
        self.call_start_time: Optional[datetime] = None
        
        # Audio output stream for PySIP
        self.audio_stream: Optional[AudioStreamAdapter] = None
        
        # Frame counters for debugging
        self._input_frame_count = 0
        self._output_frame_count = 0
        
        # OPTIMIZATION: Metrics for monitoring
        self._dropped_frame_count = 0
        self._last_drop_log_time = 0
        self._drop_log_interval = 1.0  # Log dropped frames at most once per second
        
        # Event to signal when call is fully answered and RTP ready
        self.call_answered = asyncio.Event()
        
        # Call recording
        self.enable_recording = enable_recording
        self.recording_dir = recording_dir
        self.record_separate = record_separate
        self.recorder: Optional[CallRecorder] = None
        
        # Interrupt flag for fast audio clearing
        self._interrupting = False
        
        # NEW: Queue mode for process isolation
        self._audio_in_queue = audio_in_queue
        self._audio_out_queue = audio_out_queue
        self._status_queue = status_queue
        self._queue_mode = audio_in_queue is not None and audio_out_queue is not None
        
        mode_str = "QUEUE MODE" if self._queue_mode else "DIRECT MODE"
        logger.info(f"PySIP S2S Bot initialized ({mode_str}) for user {user} on gateway {gateway}")

        # Silence detection
        self.last_activity_time = time.time()
        self.silence_threshold = 200  # RMS threshold for voice activity
        self.silence_reported = False
        # Flag to track if S2S session is still active (can be set externally)
        self._s2s_active = True
        self._silence_monitor_stopped = False
        self._silence_monitor_task = None

    async def make_call(self, destination: str):
        """Initiate outbound call.
        
        Args:
            destination: Phone number or SIP URI to call
        """
        logger.info(f"=== INITIATING CALL TO {destination} (PySIP S2S Mode - OPTIMIZED) ===")
        
        # Completely disable PySIP logging
        #import logging as pysip_logging
        #pysip_logger = pysip_logging.getLogger('PySIP')
        
        logger.info("About to create SipCall instance...")
        
        try:
            # Create SipCall instance
            self.call = SipCall(
                username=self.sip_username,
                password=self.sip_password,
                route=self.sip_server,
                callee=destination
            )
            logger.info(f"SipCall instance created for {destination}")
            
            logger.info("Registering callbacks...")
            
            # Register callbacks with error handling
            @self.call.on_call_state_changed
            async def on_state(state: CallState):
                try:
                    logger.info(f"Call state changed: {state}")
                    if state in [CallState.ENDED, CallState.FAILED, CallState.BUSY]:
                        await self._on_call_ended(state)
                except Exception as e:
                    logger.error(f"Error in on_call_state_changed callback: {e}")
                    logger.error(traceback.format_exc())
            
            @self.call.on_frame_received
            async def on_frame(frame):
                """Receive audio from phone and send to OpenAI.
                
                PySIP provides ulaw 8kHz frames (typically 160 bytes = 20ms).
                OpenAI Realtime API accepts ulaw 8kHz directly - no conversion needed!
                """
                try:
                    # Normalize frame and optional RTP timestamp
                    rtp_ts = None
                    if hasattr(frame, "data"):
                        # New-style JitterFrame from PySIP jitter buffer
                        ulaw_bytes = frame.data
                        rtp_ts = getattr(frame, "timestamp", None)
                    else:
                        # Backwards-compatible path: plain ulaw bytes
                        ulaw_bytes = frame

                    # On first frame, set up audio output stream
                    if not self.audio_stream and self.call and self.call._rtp_session:
                        self.audio_stream = AudioStreamAdapter()
                        self.call._rtp_session.set_audio_stream(self.audio_stream)
                        logger.info("Audio stream set on RTP session (triggered by first frame)")
                        
                        # Set call state flags
                        self.is_active = True
                        self.call_established = True
                        self.call_start_time = datetime.now()
                        
                        # Start silence monitor
                        self.last_activity_time = time.time()
                        self._silence_monitor_task = asyncio.create_task(self._monitor_silence())
                        logger.info("Silence monitor started")

                        # Start recording if enabled
                        if self.enable_recording:
                            # Use buffered, timestamp-aware recorder for S2S so that
                            # we can reconstruct a clean stereo call recording
                            # offline using timestamps and real silence for gaps.
                            self.recorder = S2SBufferedRecorder(
                                self.context.log_id,
                                self.recording_dir,
                                record_separate=self.record_separate,
                                record_combined=True,
                            )
                            await self.recorder.start_recording()
                        
                        # Signal that call is fully ready
                        # NOTE: This must be AFTER recorder init to avoid race condition
                        self.call_answered.set()
                        logger.info("Call fully answered and ready for audio")
                    
                    self._input_frame_count += 1
                    
                    # Calculate Audio Energy (RMS) to detect speech vs silence
                    try:
                        # Convert ulaw to 16-bit linear PCM for RMS calculation
                        pcm_data = audioop.ulaw2lin(ulaw_bytes, 2)
                        rms = audioop.rms(pcm_data, 2)
                        
                        # Only update activity time if energy is above threshold (speech detected)
                        if rms > self.silence_threshold:
                            self.last_activity_time = time.time()
                            if self.silence_reported:
                                print(f"[SILENCE DEBUG] Voice detected (RMS: {rms}), resetting silence flag")
                                self.silence_reported = False
                        
                        # Debug print every 100 frames (~2s) to help tune threshold
                        if self._input_frame_count % 100 == 0:
                            print(f"[SILENCE DEBUG] Current RMS: {rms} (Threshold: {self.silence_threshold})")
                            
                    except Exception as e:
                        print(f"[SILENCE DEBUG] Error calculating RMS: {e}")
                        # Fallback: update on every frame if calculation fails
                        self.last_activity_time = time.time()
                    
                    # Debug logging every 50 frames (~1 second)
                    if self._input_frame_count % 50 == 0:
                        logger.debug(f"Received frame #{self._input_frame_count}, size: {len(ulaw_bytes)} bytes")
                    
                    # Record incoming audio with timestamp if available
                    if self.recorder:
                        if rtp_ts is not None:
                            # Use RTP timestamp for precise placement
                            self.recorder.record_incoming_with_timestamp(ulaw_bytes, rtp_ts)
                        else:
                            # Fallback to sequential placement
                            self.recorder.record_incoming(ulaw_bytes)
                    
                    # NEW: Queue mode - put to queue instead of service call
                    if self._queue_mode:
                        try:
                            # Non-blocking put - drop frame if queue full
                            self._audio_in_queue.put_nowait(ulaw_bytes)
                        except:
                            # Queue full - drop frame (logged elsewhere)
                            pass
                    else:
                        # Original code path - send directly to OpenAI
                        await service_manager.send_s2s_audio_chunk(
                            audio_bytes=ulaw_bytes,
                            context=self.context
                        )

                except Exception as e:
                    print(f"ERROR in on_frame_received callback: {e}")
                    print(traceback.format_exc())
                    logger.error(f"Error in on_frame_received callback: {e}")
                    logger.error(traceback.format_exc())
            
            logger.info("Callbacks registered, starting PySIP call...")
            
            logger.info("Calling self.call.start()...")
            # Start the call
            await self.call.start()
            
            # This line should only be reached when call ends
            logger.info("PySIP call.start() completed")
            
        except Exception as e:
            logger.error(f"Error in make_call: {e}")
            logger.error(traceback.format_exc())
            raise
    
    async def _on_call_ended(self, state: CallState):
        """Called when call ends.
        
        Args:
            state: Final call state (ENDED, FAILED, or BUSY)
        """
        try:
            logger.info(f"=== CALL ENDED: {state} (PySIP S2S Mode - OPTIMIZED) ===")
            
            self.is_active = False
            self.call_established = False

            # Stop silence monitor
            if self._silence_monitor_task:
                self._silence_monitor_task.cancel()
                try:
                    await self._silence_monitor_task
                except asyncio.CancelledError:
                    pass
                self._silence_monitor_task = None
            
            # Stop audio stream
            if self.audio_stream:
                try:
                    self.audio_stream.input_q.put(None, block=False)
                    self.audio_stream.stream_done()
                    logger.info("Audio stream stopped")
                except Exception as e:
                    logger.warning(f"Error stopping audio stream: {e}")
            
            # Pre-mute recorder channels to avoid tail noise, then stop recording
            if self.recorder:
                try:
                    self.recorder.interrupt_outgoing()
                except Exception:
                    pass
                try:
                    self.recorder.interrupt_incoming()
                except Exception:
                    pass
                await self.recorder.stop_recording()
                self.recorder = None
            
            # Send disconnect message to agent
            await self._show_disconnected()
            
            # Clear PySIP outgoing smoothing buffer to avoid end-of-call residuals
            try:
                if self.call and hasattr(self.call, '_rtp_session') and self.call._rtp_session:
                    self.call._rtp_session.__outgoing_buffer = []
                    logger.info("Cleared PySIP outgoing buffer at call end")
            except Exception:
                pass

            # Log statistics including dropped frames
            logger.info(f"Call statistics - Input frames: {self._input_frame_count}, "
                       f"Output frames: {self._output_frame_count}, "
                       f"Dropped frames: {self._dropped_frame_count}")
            
            if self._dropped_frame_count > 0:
                drop_rate = (self._dropped_frame_count / max(1, self._output_frame_count)) * 100
                logger.warning(f"Frame drop rate: {drop_rate:.2f}% ({self._dropped_frame_count}/{self._output_frame_count})")
            
        except Exception as e:
            logger.error(f"Error in _on_call_ended: {e}")
            logger.error(traceback.format_exc())
    
    async def send_tts_audio(self, audio_chunk: bytes, timestamp=None):
        """Send TTS audio chunk to the SIP call - OPTIMIZED NON-BLOCKING VERSION.
        
        This is the REQUIRED interface called by the session manager.
        OpenAI sends ulaw 8kHz audio which we pass through directly.
        
        OPTIMIZATION: Uses non-blocking put_nowait() to prevent RTP disruption.
        
        NOTE: In queue mode, this method is called by the audio queue reader task
        in the subprocess. The audio comes from the audio_out_queue which is fed
        by the main process. In direct mode, this is called directly by the
        session manager.
        Dropped frames are logged as CRITICAL since they directly impact audio quality.
        
        Args:
            audio_chunk: Audio data from OpenAI (ulaw 8kHz)
            timestamp: Optional timestamp when this audio should start playing
        """
        try:
            if not self.is_active:
                logger.warning(f"Cannot send audio - call not active (is_active={self.is_active})")
                return
            
            if not self.audio_stream:
                logger.warning("Cannot send audio - audio stream not initialized")
                return

            # Update activity timestamp (Output)
            # Project activity to when this audio will FINISH playing
            # This handles bursting by chaining consecutive chunks
            chunk_duration = len(audio_chunk) / 8000.0
            self.last_activity_time = max(self.last_activity_time, time.time()) + chunk_duration
            
            if self.silence_reported:
                self.silence_reported = False
            
            # Split into 160-byte frames
            FRAME_SIZE = 160
            
            # Check interrupt flag before processing chunk
            if self._interrupting:
                return  # Abort immediately on interrupt
            
            # Prepare all frames with timestamps first (batching)
            frames_to_send = []
            for i in range(0, len(audio_chunk), FRAME_SIZE):
                frame = audio_chunk[i : i + FRAME_SIZE]
                frame_timestamp = timestamp + (i / 8000.0) if timestamp else None

                # Add to batch (send all frames, even partial ones at end)
                frames_to_send.append((frame, frame_timestamp))
            
            # Now send all frames in batch without yielding
            # This is more efficient and keeps queue fuller
            for frame, frame_timestamp in frames_to_send:
                # Check interrupt flag before each put
                if self._interrupting:
                    return  # Abort immediately on interrupt
                
                try:
                    # BLOCKING PUT: Ensures frames are never dropped
                    # This prevents queue from emptying which causes clicks/pops
                    # Timeout of 0.5s is generous - should never be reached in normal operation
                    if frame_timestamp:
                        self.audio_stream.input_q.put(
                            (frame, frame_timestamp), block=True, timeout=0.5
                        )
                    else:
                        self.audio_stream.input_q.put(frame, block=True, timeout=0.5)

                    # Record AFTER successful queueing. For the buffered
                    # recorder we pass the per-frame timestamp so it can
                    # place audio precisely on the output timeline.
                    if self.recorder:
                        try:
                            # S2SBufferedRecorder accepts an optional timestamp
                            self.recorder.record_outgoing(frame, timestamp=frame_timestamp)
                        except TypeError:
                            # Fallback for any legacy recorder that only takes
                            # the audio bytes.
                            self.recorder.record_outgoing(frame)
                    self._output_frame_count += 1
                    
                except queue.Full:
                    # This should never happen with blocking put and reasonable timeout
                    # But if it does, log it as critical
                    logger.critical(f"Audio queue full even with blocking put! Queue size: {self.audio_stream.input_q.qsize()}")
                    self._dropped_frame_count += 1
                    continue
                    
                except Exception as e:
                    logger.error(f"Unexpected error queueing frame: {e}")
                    break
        except Exception as e:
            logger.error(f"Error in send_tts_audio: {e}")
            logger.error(traceback.format_exc())
    
    def clear_audio_queue(self):
        """Clear all queued audio frames (for interruption)."""
        try:
            # Set interrupt flag to stop ongoing sends
            self._interrupting = True

            # Reset activity timer to now (cancelling future projected time)
            self.last_activity_time = time.time()

            # Reset recorder's outgoing hold to silence to avoid buzz on combined file
            try:
                if self.recorder:
                    self.recorder.interrupt_outgoing()
            except Exception as _e:
                logger.debug(f"interrupt_outgoing safe-ignored: {_e}")
            
            if self.audio_stream:
                try:
                    # Drain the main queue
                    cleared_count = 0
                    while not self.audio_stream.input_q.empty():
                        try:
                            self.audio_stream.input_q.get_nowait()
                            cleared_count += 1
                        except:
                            break
                    logger.info(f"Cleared {cleared_count} audio frames from main queue")
                except Exception as e:
                    logger.error(f"Error clearing main audio queue: {e}")
            
            # Clear PySIP jitter buffer
            if self.call and hasattr(self.call, '_rtp_session') and self.call._rtp_session:
                self.call._rtp_session.__outgoing_buffer = []
                logger.info("Cleared jitter buffer")
        finally:
            # Always reset interrupt flag, even on error
            self._interrupting = False
    
    async def hangup_call(self):
        """Initiate call hangup and cleanup."""
        try:
            logger.info("Hangup requested. Performing cleanup...")
            
            if self.call:
                await self.call.stop("Agent hangup")
                logger.info("Call stop requested")
            else:
                logger.warning("No active call to hang up")
                
        except Exception as e:
            logger.error(f"Error in hangup_call: {e}")
            logger.error(traceback.format_exc())
    
    async def _monitor_silence(self):
        """Monitor for silence on both channels."""
        try:
            while self.is_active:
                await asyncio.sleep(0.5)
                duration = time.time() - self.last_activity_time
                # print(f"[SILENCE DEBUG] Duration since last activity: {duration:.2f}s")
                
                # Trigger if > 4.0s silence and not yet reported for this gap
                if duration > 10.0 and not self.silence_reported:
                    self.silence_reported = True
                    msg = f"[SYSTEM: No audio detected on either channel for {duration:.1f} seconds. (RMS Threshold: {self.silence_threshold})]"
                    logger.info(f"Silence detected: {msg}")
                    
                    if self._queue_mode and self._status_queue:
                        print(f"[SILENCE DEBUG] Silence detected ({duration:.1f}s), sending event to main process")
                        # Process Isolation: Send event to main process
                        try:
                            self._status_queue.put({
                                'type': 'silence_timeout',
                                'duration': duration,
                                'message': msg,
                                'timestamp': datetime.now().isoformat()
                            })
                        except Exception as e:
                            logger.error(f"Error sending silence event: {e}")
                    else:
                        print(f"[SILENCE DEBUG] Silence detected ({duration:.1f}s), injecting message directly")
                        # Direct Mode: Inject message directly
                        try:
                            # Use send_s2s_message directly to avoid local logging (let echo handle it)
                            if hasattr(self.context, 'send_s2s_message'):
                                payload = { "role": "user", "content": [ { "type": "text", "text": msg} ] }
                                await self.context.send_s2s_message(payload)
                            else:
                                await self._inject_system_message(msg)
                        except Exception as e:
                            # S2S session likely closed - stop monitoring
                            logger.warning(f"Failed to send silence notification (S2S likely closed): {e}")
                            self._s2s_active = False
                            break
        except asyncio.CancelledError:
            logger.info("Silence monitor cancelled")
        except Exception as e:
            logger.error(f"Error in silence monitor: {e}")
            logger.error(traceback.format_exc())

    async def _inject_system_message(self, msg: str):
        """Helper to inject system message in Direct Mode."""
        try:
            await service_manager.backend_user_message(message=msg, context=self.context)
            await service_manager.send_message_to_agent(
                session_id=self.context.log_id,
                message=msg,
                context=self.context
            )
        except Exception as e:
            logger.error(f"Error injecting system message: {e}")

    async def _show_disconnected(self):
        """Send disconnect message to agent."""
        try:
            msg = "\n\nSYSTEM: -- CALL DISCONNECTED --\n\n"
            
            if self._queue_mode and self._status_queue:
                # Process Isolation: Send event to main process
                try:
                    self._status_queue.put({
                        'type': 'call_disconnected',
                        'message': msg,
                        'timestamp': datetime.now().isoformat()
                    })
                except Exception as e:
                    logger.error(f"Error sending disconnect event: {e}")
            else:
                # Direct Mode: Log directly
                # 1. Update UI
                await service_manager.backend_user_message(message=msg, context=self.context)
                # 2. Send to Agent (so it knows call ended) and Chat Log
                await service_manager.send_message_to_agent(
                    session_id=self.context.log_id,
                    message=msg,
                    context=self.context
                )
            
            logger.info("Disconnect message sent to agent")
            
        except Exception as e:
            logger.error(f"Error sending disconnect message: {e}")
            logger.error(traceback.format_exc())
    
    def hang(self):
        """Synchronous hangup method for compatibility.
        
        This is called by session manager cleanup code.
        """
        try:
            if self.call:
                # Schedule async hangup
                try:
                    loop = asyncio.get_event_loop()
                    if not loop.is_closed():
                        asyncio.create_task(self.hangup_call())
                        logger.info("Async hangup scheduled")
                except Exception as e:
                    logger.error(f"Error scheduling hangup: {e}")
                    logger.error(traceback.format_exc())
            else:
                logger.warning("hang() called but no active call")
                
        except Exception as e:
            logger.error(f"Error in hang(): {e}")
            logger.error(traceback.format_exc())
    
    def get_metrics(self) -> dict:
        """Get audio metrics for monitoring.
        
        Returns:
            dict: Metrics including frame counts and drop rate
        """
        total_frames = max(1, self._output_frame_count)
        drop_rate = (self._dropped_frame_count / total_frames) * 100
        
        return {
            "input_frames": self._input_frame_count,
            "output_frames": self._output_frame_count,
            "dropped_frames": self._dropped_frame_count,
            "drop_rate_percent": drop_rate,
            "is_active": self.is_active,
            "call_established": self.call_established
        }

    async def _wait_for_call_end(self):
        """Wait for call to end (used by subprocess).
        
        This is called by the subprocess wrapper to keep the call alive
        until it ends naturally or is hung up.
        """
        while self.is_active or self.call_established:
            await asyncio.sleep(0.5)
        logger.info("Call ended, exiting wait loop")
