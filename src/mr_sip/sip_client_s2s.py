#!/usr/bin/env python3
"""
SIP Client for Speech-to-Speech Mode

This client variant is designed for use with OpenAI Realtime API or other
speech-to-speech systems. It only handles audio INPUT - capturing from JACK
and sending to the S2S system. Audio OUTPUT is handled by the agent calling
sip_audio_out_chunk.

Key differences from sip_client_v2.py:
- No STT provider setup (S2S handles this)
- No transcription callbacks (S2S handles this)
- No audio output routing (agent handles this)
- Just captures audio and sends to send_s2s_audio_chunk service
"""

import os
import time
import asyncio
import logging
import numpy as np
from baresipy import BareSIP
from datetime import datetime
from pathlib import Path
from lib.providers.services import service_manager
from .audio_handler import AudioHandler
from .audio.jack_input_capture import JACKAudioCapture

logger = logging.getLogger(__name__)

class MindRootSIPBotS2S(BareSIP):
    """SIP phone bot for Speech-to-Speech mode.
    
    This client only handles audio INPUT from the phone call.
    Audio OUTPUT is handled by the agent via sip_audio_out_chunk.
    """
    
    def __init__(self, user, password, gateway, audio_dir=".", context=None):
        """
        Args:
            user: SIP username
            password: SIP password
            gateway: SIP gateway
            audio_dir: Directory for audio files
            context: MindRoot ChatContext
        """
        # Set up audio directory
        self.audio_dir = audio_dir or os.path.expanduser("~/.baresip")
         
        # Initialize baresipy
        super().__init__(user, password, gateway, block=False)
        
        # MindRoot integration
        self.context = context
        
        # Call tracking
        self.call_start_time = None

        self.audio_ready = False

        # Audio processing
        self.audio_handler = AudioHandler()
        self.audio_capture = None
        
        # Store reference to main event loop
        try:
            self.main_loop = asyncio.get_running_loop()
        except RuntimeError:
            self.main_loop = None
            
    def handle_call_established(self):
        """When call connects, setup JACK and start audio capture."""
        logger.info("=== CALL ESTABLISHED (S2S Mode) ===")
        logger.info(f"S2S_DEBUG: Main loop available: {self.main_loop is not None}")
        logger.info(f"S2S_DEBUG: Main loop closed: {self.main_loop.is_closed() if self.main_loop else 'N/A'}")
        self.call_start_time = datetime.now()
        
        # Setup JACK audio output
        if not self.audio_handler.jack_enabled:
            self.audio_handler.setup_jack_audio()
            time.sleep(0.1)
        
        # Configure baresip to use JACK FIRST
        self.audio_handler.configure_baresip_jack(self)
        
        # Wait a bit for baresip to activate its JACK client
        logger.info("S2S_DEBUG: Waiting for baresip JACK client to activate...")
        time.sleep(0.1)
            
        # Connect JACK ports
        # Retry connection a few times since baresip may take time to activate
        ok = False
        for attempt in range(5):
            ok = self.audio_handler.connect_jack_to_baresip()
            if ok:
                break
            logger.warning(f"S2S_DEBUG: JACK connection attempt {attempt+1} failed, retrying...")
            time.sleep(0.1)
        
        if not ok:
            logger.error("S2S_DEBUG: Failed to connect JACK ports after 5 attempts")
            self._schedule_coroutine(self.hangup_call())
            return

        # Setup audio capture
        self._schedule_coroutine(self._setup_audio_capture())
        time.sleep(0.1)
        self.audio_ready = True
        
    async def _setup_audio_capture(self):
        """Setup JACK audio capture to send to S2S system."""
        try:
            logger.info("S2S_DEBUG: _setup_audio_capture called")
            logger.info("Starting JACK audio capture for S2S mode...")
            
            # Create JACK audio capture
            # Target 24kHz for OpenAI (will be resampled from JACK rate)
            self.audio_capture = JACKAudioCapture(
                target_sample_rate=24000,  # OpenAI expects 24kHz
                chunk_duration_s=0.1,
                chunk_callback=self._on_audio_chunk_from_jack,
                stereo_mix=True,
                agc_target_rms=0.15,
                agc_max_gain=20.0
            )
            logger.info("S2S_DEBUG: JACKAudioCapture created")
            
            await self.audio_capture.start()
            logger.info("S2S_DEBUG: Audio capture started successfully")
            logger.info(f"S2S_DEBUG: Capture running: {self.audio_capture._running if hasattr(self.audio_capture, '_running') else 'unknown'}")
            
        except Exception as e:
            logger.error(f"Error setting up audio capture: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
    async def _on_audio_chunk_from_jack(self, audio_chunk: np.ndarray):
        """Callback for audio chunks from JACK - send to S2S system."""
        try:
            #return
            if not hasattr(self, '_audio_chunk_count'):
                self._audio_chunk_count = 0
            self._audio_chunk_count += 1
            
            # Convert numpy array to bytes (already at 24kHz from JACKAudioCapture)
            # OpenAI expects PCM 16-bit
            #if self._audio_chunk_count % 50 == 0:
            #    logger.info(f"S2S_DEBUG: Audio input chunk #{self._audio_chunk_count}, size: {len(audio_chunk)}")
            
            audio_bytes = (audio_chunk * 32767).astype(np.int16).tobytes()
            #logger.debug(f"S2S_DEBUG: Converted to {len(audio_bytes)} bytes PCM")
                
            # Send to S2S system (OpenAI or other provider)
            #logger.debug(f"S2S_DEBUG: Calling send_s2s_audio_chunk with context.log_id={self.context.log_id if self.context else None}")
            
            await service_manager.send_s2s_audio_chunk(
                audio_bytes=audio_bytes,
                context=self.context
            )
            
            #if self._audio_chunk_count % 50 == 0:
            #    logger.info(f"S2S_DEBUG: Successfully sent chunk #{self._audio_chunk_count} to S2S")
                
        except Exception as e:
            logger.error(f"Error sending audio to S2S system: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
    async def send_tts_audio(self, audio_chunk: bytes):
        """Send TTS audio chunk to the SIP call via JACK.
        
        This is called by the session manager when audio needs to be sent to the phone.
        OpenAI sends PCM 16-bit at 24kHz, but we need to convert it for JACK.
        """
        try:
            # OpenAI sends PCM 16-bit at 24kHz
            # Convert to format expected by audio_handler
            import numpy as np
            audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
            # audio_handler expects the same format, so we can pass through
            await self.audio_handler.send_tts_audio(audio_chunk)
            logger.debug(f"Sent {len(audio_chunk)} bytes TTS audio to JACK")
        except Exception as e:
            logger.error(f"Error sending TTS audio: {e}")
            
    async def hangup_call(self):
        """Initiate call hangup and cleanup."""
        logger.info("Hangup requested. Performing cleanup...")
        self.handle_call_ended("Hangup command received")
        self.hang()
        
    def handle_call_ended(self, reason):
        """When call ends, cleanup resources."""
        logger.info("=== CALL ENDED (S2S Mode) ===")
        
        # Stop audio capture
        if self.audio_capture:
            self._schedule_coroutine(self.audio_capture.stop())
            
        # Cleanup audio handler
        self.audio_handler.cleanup(self)
        
        # Show disconnect message - but only if we have a valid event loop
        try:
            if self.main_loop and not self.main_loop.is_closed():
                self._schedule_coroutine(self.show_disconnected())
            else:
                logger.warning("Cannot send disconnect message - no event loop available")
        except Exception as e:
            logger.error(f"Error scheduling disconnect message: {e}")
        
    async def show_disconnected(self):
        """Send disconnect message to agent."""
        msg = "\n\nSYSTEM: -- CALL DISCONNECTED --\n\n"
        await service_manager.backend_user_message(message=msg)
        await service_manager.send_message_to_agent(
            session_id=self.context.log_id,
            message=msg,
            context=self.context
        )
        
    def _schedule_coroutine(self, coro):
        """Schedule a coroutine to run in the main event loop."""
        if self.main_loop and not self.main_loop.is_closed():
            try:
                return asyncio.run_coroutine_threadsafe(coro, self.main_loop)
            except Exception as e:
                logger.error(f"Failed to schedule coroutine: {e}")
        return None
