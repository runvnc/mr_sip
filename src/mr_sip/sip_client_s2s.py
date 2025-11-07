#!/usr/bin/env python3
"""
SIP Client for Speech-to-Speech Mode using PySIP

This client uses PySIP library instead of baresip/JACK for SIP call handling.
It handles both audio input (phone -> OpenAI) and output (OpenAI -> phone).

Key features:
- No JACK dependencies
- No baresip dependencies  
- Direct ulaw 8kHz audio (no conversion needed for OpenAI Realtime API)
- Async/await based architecture
- Preserves send_tts_audio() interface for session manager compatibility
"""

import asyncio
import logging
import queue
from datetime import datetime
from typing import Optional
from PySIP import SipCall
from PySIP.filters import CallState
from lib.providers.services import service_manager

logger = logging.getLogger(__name__)

class MindRootSIPBotS2S:
    """SIP phone bot for Speech-to-Speech mode using PySIP.
    
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
        self.sip_server = gateway
        self.context = context
        
        # Call state tracking
        self.call: Optional[SipCall] = None
        self.is_active = False
        self.call_established = False
        self.call_start_time: Optional[datetime] = None
        
        # Audio output queue for sending to phone
        self.audio_output_queue: Optional[queue.Queue] = None
        self._audio_sender_task: Optional[asyncio.Task] = None
        
        # Frame counters for debugging
        self._input_frame_count = 0
        self._output_frame_count = 0
        
    async def make_call(self, destination: str):
        """Initiate outbound call.
        
        Args:
            destination: Phone number or SIP URI to call
        """
        logger.info(f"=== INITIATING CALL TO {destination} (PySIP S2S Mode) ===")
        
        # Create SipCall instance
        self.call = SipCall(
            username=self.sip_username,
            password=self.sip_password,
            route=self.sip_server,
            callee=destination
        )
        
        # Register callbacks
        @self.call.on_call_state_changed
        async def on_state(state: CallState):
            logger.info(f"Call state changed: {state}")
            if state == CallState.ANSWERED:
                await self._on_call_answered()
            elif state in [CallState.ENDED, CallState.FAILED, CallState.BUSY]:
                await self._on_call_ended(state)
        
        @self.call.on_frame_received
        async def on_frame(frame: bytes):
            """Receive audio from phone and send to OpenAI.
            
            PySIP provides ulaw 8kHz frames (typically 160 bytes = 20ms).
            OpenAI Realtime API accepts ulaw 8kHz directly - no conversion needed!
            """
            self._input_frame_count += 1
            
            # Debug logging every 50 frames (~1 second)
            if self._input_frame_count % 50 == 0:
                logger.debug(f"Received frame #{self._input_frame_count}, size: {len(frame)} bytes")
            
            try:
                # Send directly to OpenAI S2S system
                await service_manager.send_s2s_audio_chunk(
                    audio_bytes=frame,
                    context=self.context
                )
            except Exception as e:
                logger.error(f"Error sending audio to S2S system: {e}")
        
        # Start the call
        logger.info("Starting PySIP call...")
        await self.call.start()
    
    async def _on_call_answered(self):
        """Called when call connects and is answered."""
        logger.info("=== CALL ANSWERED (PySIP S2S Mode) ===")
        
        self.is_active = True
        self.call_established = True
        self.call_start_time = datetime.now()
        
        # Create queue for audio output (OpenAI -> phone)
        self.audio_output_queue = queue.Queue(maxsize=50)  # ~1 second buffer
        
        # Start audio output sender task
        self._audio_sender_task = asyncio.create_task(self._audio_output_loop())
        logger.info("Audio output loop started")
    
    async def _on_call_ended(self, state: CallState):
        """Called when call ends.
        
        Args:
            state: Final call state (ENDED, FAILED, or BUSY)
        """
        logger.info(f"=== CALL ENDED: {state} (PySIP S2S Mode) ===")
        
        self.is_active = False
        self.call_established = False
        
        # Stop audio output
        if self.audio_output_queue:
            try:
                self.audio_output_queue.put(None, block=False)
            except:
                pass
        
        if self._audio_sender_task:
            self._audio_sender_task.cancel()
            try:
                await self._audio_sender_task
            except asyncio.CancelledError:
                pass
        
        # Send disconnect message to agent
        await self._show_disconnected()
        
        # Log statistics
        logger.info(f"Call statistics - Input frames: {self._input_frame_count}, Output frames: {self._output_frame_count}")
    
    async def _audio_output_loop(self):
        """Background task that sends audio from queue to phone via RTP.
        
        This reads from audio_output_queue and feeds frames to PySIP's RTP session.
        """
        logger.info("Audio output loop starting...")
        
        try:
            while self.is_active:
                try:
                    # Get audio chunk from queue (with timeout)
                    audio_chunk = await asyncio.wait_for(
                        asyncio.to_thread(self.audio_output_queue.get, timeout=1.0),
                        timeout=2.0
                    )
                    
                    if audio_chunk is None:  # Sentinel to stop
                        logger.info("Received stop signal in audio output loop")
                        break
                    
                    # Send to RTP session
                    await self._send_to_rtp(audio_chunk)
                    self._output_frame_count += 1
                    
                    if self._output_frame_count % 50 == 0:
                        logger.debug(f"Sent frame #{self._output_frame_count} to RTP, queue size: {self.audio_output_queue.qsize()}")
                    
                except queue.Empty:
                    continue
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Error in audio output loop: {e}")
                    await asyncio.sleep(0.1)
                    
        except asyncio.CancelledError:
            logger.info("Audio output loop cancelled")
        finally:
            logger.info("Audio output loop exiting")
    
    async def _send_to_rtp(self, audio_chunk: bytes):
        """Send audio chunk to PySIP RTP session.
        
        Args:
            audio_chunk: Audio data to send (ulaw 8kHz)
        """
        if not self.call or not self.call._rtp_session:
            logger.warning("Cannot send audio - no RTP session")
            return
        
        # PySIP expects 160-byte frames for 8kHz ulaw (20ms)
        # Chunk the data if needed
        FRAME_SIZE = 160
        
        for i in range(0, len(audio_chunk), FRAME_SIZE):
            frame = audio_chunk[i:i+FRAME_SIZE]
            
            # Only send complete frames
            if len(frame) == FRAME_SIZE:
                try:
                    # Put frame in RTP session's input queue
                    await asyncio.to_thread(
                        self.call._rtp_session._input_queue.put_nowait,
                        frame
                    )
                except Exception as e:
                    logger.error(f"Failed to queue RTP frame: {e}")
    
    async def send_tts_audio(self, audio_chunk: bytes):
        """Send TTS audio chunk to the SIP call.
        
        This is the REQUIRED interface called by the session manager.
        OpenAI sends ulaw 8kHz audio which we pass through directly.
        
        Args:
            audio_chunk: Audio data from OpenAI (ulaw 8kHz)
        """
        if not self.is_active or not self.audio_output_queue:
            logger.warning("Cannot send audio - call not active")
            return
        
        try:
            # Queue audio for sending to phone
            # Use non-blocking put to avoid delays
            self.audio_output_queue.put_nowait(audio_chunk)
        except queue.Full:
            logger.warning("Audio output queue full, dropping chunk to prevent latency buildup")
        except Exception as e:
            logger.error(f"Failed to queue audio: {e}")
    
    async def hangup_call(self):
        """Initiate call hangup and cleanup."""
        logger.info("Hangup requested. Performing cleanup...")
        
        if self.call:
            await self.call.stop("Agent hangup")
    
    async def _show_disconnected(self):
        """Send disconnect message to agent."""
        msg = "\n\nSYSTEM: -- CALL DISCONNECTED --\n\n"
        
        try:
            await service_manager.backend_user_message(message=msg)
            await service_manager.send_message_to_agent(
                session_id=self.context.log_id,
                message=msg,
                context=self.context
            )
        except Exception as e:
            logger.error(f"Error sending disconnect message: {e}")
    
    def hang(self):
        """Synchronous hangup method for compatibility.
        
        This is called by session manager cleanup code.
        """
        if self.call:
            # Schedule async hangup
            try:
                loop = asyncio.get_event_loop()
                if not loop.is_closed():
                    asyncio.create_task(self.hangup_call())
            except Exception as e:
                logger.error(f"Error scheduling hangup: {e}")
