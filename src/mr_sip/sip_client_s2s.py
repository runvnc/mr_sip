#!/usr/bin/env python3
"""
SIP Client for Speech-to-Speech Mode using PySIP

This client uses PySIP library instead of baresip/JACK for SIP call handling.
It handles OUTBOUND calls only (not incoming calls).
Handles both audio input (phone -> OpenAI) and output (OpenAI -> phone).

Key features:
- No JACK dependencies
- No baresip dependencies  
- Direct ulaw 8kHz audio (no conversion needed for OpenAI Realtime API)
- Async/await based architecture
- Preserves send_tts_audio() interface for session manager compatibility
- Outbound calls only (no incoming call support)
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

logger = logging.getLogger(__name__)

class AudioStreamAdapter:
    """Adapter to feed audio to PySIP's RTP session.
    
    PySIP expects an object with an input_q attribute (queue.Queue)
    that it reads audio frames from.
    """
    def __init__(self):
        self.input_q = queue.Queue(maxsize=50)
        self.stream_id = "tts_output"
        self._done = False
    
    def stream_done(self):
        """Mark stream as done."""
        self._done = True

class MindRootSIPBotS2S:
    """SIP phone bot for Speech-to-Speech mode using PySIP.
    
    OUTBOUND CALLS ONLY - Does not handle incoming calls.
    
    Handles bidirectional audio:
    - Input: Phone audio -> OpenAI (via on_frame_received callback)
    - Output: OpenAI audio -> Phone (via send_tts_audio method)
    """
    
    def __init__(self, user: str, password: str, gateway: str, audio_dir: str = ".", context=None):
        """
        Args:
            user: SIP username
            password: SIP password
            gateway: SIP gateway (format: "host:port")
            audio_dir: Unused, kept for compatibility
            context: MindRoot ChatContext
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
        
        logger.info(f"PySIP S2S Bot initialized for user {user} on gateway {gateway}")
        
    async def make_call(self, destination: str):
        """Initiate outbound call.
        
        Args:
            destination: Phone number or SIP URI to call
        """
        logger.info(f"=== INITIATING CALL TO {destination} (PySIP S2S Mode) ===")
        
        # Enable PySIP debug logging to see SIP messages
        import logging as pysip_logging
        pysip_logging.getLogger('PySIP').setLevel(pysip_logging.DEBUG)
        
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
                    if state == CallState.ANSWERED:
                        await asyncio.sleep(1)
                        await self._on_call_answered()
                    elif state in [CallState.ENDED, CallState.FAILED, CallState.BUSY]:
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
                    self._input_frame_count += 1
                    
                    # Debug logging every 50 frames (~1 second)
                    if self._input_frame_count % 50 == 0:
                        logger.debug(f"Received frame #{self._input_frame_count}, size: {len(frame)} bytes")
                    
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
    
    async def _on_call_answered(self):
        """Called when call connects and is answered."""
        try:
            logger.info("=== CALL ANSWERED (PySIP S2S Mode) ===")
            
            self.is_active = True
            self.call_established = True
            self.call_start_time = datetime.now()
            
            # Create audio stream adapter for output
            self.audio_stream = AudioStreamAdapter()
            logger.info("Audio stream adapter created")
            
            # Wait a bit for RTP session to be fully initialized
            await asyncio.sleep(0.5)
            
            # Set the audio stream on the RTP session
            if self.call and self.call._rtp_session:
                self.call._rtp_session.set_audio_stream(self.audio_stream)
                logger.info("Audio stream set on RTP session")
            else:
                logger.warning("RTP session not available yet")
                # Try again after a longer delay
                await asyncio.sleep(0.5)
                if self.call and self.call._rtp_session:
                    self.call._rtp_session.set_audio_stream(self.audio_stream)
                    logger.info("Audio stream set on RTP session (retry)")
                else:
                    logger.error("RTP session still not available after retry")
                    
        except Exception as e:
            logger.error(f"Error in _on_call_answered: {e}")
            logger.error(traceback.format_exc())
    
    async def _on_call_ended(self, state: CallState):
        """Called when call ends.
        
        Args:
            state: Final call state (ENDED, FAILED, or BUSY)
        """
        try:
            logger.info(f"=== CALL ENDED: {state} (PySIP S2S Mode) ===")
            
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
            
            # Send disconnect message to agent
            await self._show_disconnected()
            
            # Log statistics
            logger.info(f"Call statistics - Input frames: {self._input_frame_count}, Output frames: {self._output_frame_count}")
            
        except Exception as e:
            logger.error(f"Error in _on_call_ended: {e}")
            logger.error(traceback.format_exc())
    
    async def send_tts_audio(self, audio_chunk: bytes):
        """Send TTS audio chunk to the SIP call.
        
        This is the REQUIRED interface called by the session manager.
        OpenAI sends ulaw 8kHz audio which we pass through directly.
        
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
            
            # PySIP expects 160-byte frames for 8kHz ulaw (20ms)
            # Chunk the data if needed
            FRAME_SIZE = 80 #160
            
            for i in range(0, len(audio_chunk), FRAME_SIZE):
                frame = audio_chunk[i:i+FRAME_SIZE]
                
                # Only send complete frames
                if len(frame) == FRAME_SIZE:
                    try:
                        self.audio_stream.input_q.put_nowait(frame)
                        self._output_frame_count += 1
                        
                        if self._output_frame_count % 50 == 0:
                            logger.debug(f"Queued frame #{self._output_frame_count}, queue size: {self.audio_stream.input_q.qsize()}")
                    except queue.Full:
                        logger.warning("Audio queue full, dropping frame to prevent latency buildup")
                else:
                    # Log incomplete frames for debugging
                    if len(frame) > 0:
                        logger.debug(f"Skipping incomplete frame of {len(frame)} bytes (expected {FRAME_SIZE})")
                        
        except Exception as e:
            logger.error(f"Error in send_tts_audio: {e}")
            logger.error(traceback.format_exc())
    
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
