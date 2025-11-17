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
from datetime import datetime
from typing import Optional
from PySIP.sip_call import SipCall
from PySIP.filters import CallState
from lib.providers.services import service_manager
from .call_recorder import CallRecorder

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
                 record_separate: bool = False):
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
        
        logger.info(f"PySIP S2S Bot initialized (OPTIMIZED) for user {user} on gateway {gateway}")
        
    async def make_call(self, destination: str):
        """Initiate outbound call.
        
        Args:
            destination: Phone number or SIP URI to call
        """
        logger.info(f"=== INITIATING CALL TO {destination} (PySIP S2S Mode - OPTIMIZED) ===")
        
        # Completely disable PySIP logging
        import logging as pysip_logging
        pysip_logger = pysip_logging.getLogger('PySIP')
        
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
            async def on_frame(frame: bytes):
                """Receive audio from phone and send to OpenAI.
                
                PySIP provides ulaw 8kHz frames (typically 160 bytes = 20ms).
                OpenAI Realtime API accepts ulaw 8kHz directly - no conversion needed!
                """
                try:
                    # On first frame, set up audio output stream
                    if not self.audio_stream and self.call and self.call._rtp_session:
                        self.audio_stream = AudioStreamAdapter()
                        self.call._rtp_session.set_audio_stream(self.audio_stream)
                        logger.info("Audio stream set on RTP session (triggered by first frame)")
                        
                        # Signal that call is fully ready
                        self.call_answered.set()
                        logger.info("Call fully answered and ready for audio")
                        
                        # Start recording if enabled
                        if self.enable_recording:
                            self.recorder = CallRecorder(self.context.log_id, self.recording_dir, 
                                                        record_separate=self.record_separate, record_combined=True)
                            await self.recorder.start_recording()
                    
                    self._input_frame_count += 1
                    
                    # Debug logging every 50 frames (~1 second)
                    if self._input_frame_count % 50 == 0:
                        logger.debug(f"Received frame #{self._input_frame_count}, size: {len(frame)} bytes")
                    
                    # Record incoming audio (non-blocking, already optimized)
                    if self.recorder:
                        self.recorder.record_incoming(frame)
                    
                    # Send directly to OpenAI S2S system
                    await service_manager.send_s2s_audio_chunk(
                        audio_bytes=frame,
                        context=self.context
                    )
                except Exception as e:
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
    
    async def send_tts_audio(self, audio_chunk: bytes):
        """Send TTS audio chunk to the SIP call - OPTIMIZED NON-BLOCKING VERSION.
        
        This is the REQUIRED interface called by the session manager.
        OpenAI sends ulaw 8kHz audio which we pass through directly.
        
        OPTIMIZATION: Uses non-blocking put_nowait() to prevent RTP disruption.
        Dropped frames are logged as CRITICAL since they directly impact audio quality.
        
        Args:
            audio_chunk: Audio data from OpenAI (ulaw 8kHz)
        """
        try:
            if not self.is_active:
                logger.warning(f"Cannot send audio - call not active (is_active={self.is_active})")
                return
            
            if not self.audio_stream:
                logger.warning("Cannot send audio - audio stream not initialized")
                return
            
            # Split into 160-byte frames for frame-by-frame queueing
            # This allows interruption by clearing the queue
            FRAME_SIZE = 160
            
            # Check interrupt flag before processing chunk
            if self._interrupting:
                return  # Abort immediately on interrupt
            
            for i in range(0, len(audio_chunk), FRAME_SIZE):
                frame = audio_chunk[i:i+FRAME_SIZE]
                
                # Only send complete frames
                if len(frame) == FRAME_SIZE:
                    try:
                        # Check interrupt flag BEFORE attempting to queue
                        if self._interrupting:
                            return  # Abort immediately
                        
                        # OPTIMIZATION: Non-blocking put - drop frame if queue is full
                        # This prevents blocking the audio path and disrupting RTP
                        self.audio_stream.input_q.put_nowait(frame)
                        
                        # Record AFTER successful queueing
                        if self.recorder:
                            self.recorder.record_outgoing(frame)
                        self._output_frame_count += 1
                        
                    except queue.Full:
                        # CRITICAL: Frame dropped due to full queue
                        # This directly impacts audio quality
                        self._dropped_frame_count += 1
                        
                        # Rate-limited logging to avoid log spam
                        import time
                        current_time = time.time()
                        if current_time - self._last_drop_log_time >= self._drop_log_interval:
                            logger.critical(
                                f"AUDIO FRAME DROPPED! Queue full. "
                                f"Total dropped: {self._dropped_frame_count}, "
                                f"Drop rate: {(self._dropped_frame_count / max(1, self._output_frame_count)) * 100:.2f}%"
                            )
                            self._last_drop_log_time = current_time
                        
                        # Don't record dropped frames
                        continue
                        
        except Exception as e:
            logger.error(f"Error in send_tts_audio: {e}")
            logger.error(traceback.format_exc())
    
    def clear_audio_queue(self):
        """Clear all queued audio frames (for interruption)."""
        try:
            # Set interrupt flag to stop ongoing sends
            self._interrupting = True
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
    
    async def _show_disconnected(self):
        """Send disconnect message to agent."""
        try:
            msg = "\n\nSYSTEM: -- CALL DISCONNECTED --\n\n"
            
            await service_manager.backend_user_message(message=msg)
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
