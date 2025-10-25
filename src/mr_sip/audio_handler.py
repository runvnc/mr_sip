#!/usr/bin/env python3
"""
Audio Handler for MindRoot SIP Plugin
Handles audio processing, format conversion, and JACK integration
"""

import audioop
import numpy as np
import logging
from scipy import signal
from .jack_streamer import JACKAudioStreamer
import traceback
logger = logging.getLogger(__name__)

class AudioHandler:
    """Handles audio processing and JACK streaming for SIP calls."""
    
    def __init__(self):
        self.jack_streamer = None
        self.jack_enabled = False
        
    def setup_jack_audio(self):
        """Initialize JACK audio streaming."""
        try:
            self.jack_streamer = JACKAudioStreamer()
            self.jack_streamer.start()
            self.jack_enabled = True
            logger.info("JACK audio setup complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup JACK audio: {e}")
            return False
    
    def configure_baresip_jack(self, baresip_bot):
        """Configure baresip to use JACK for both input (ausrc) and output (auplay).

        Let baresip create its own JACK ports, then we connect to them separately.
        """
        if baresip_bot:
            try:
                if os.environ.get("BARESIP_JACK_V", "0") == "1":
                    # Ensure JACK module is loaded in baresip
                    baresip_bot.do_command("/module_load jack")
                    # Use JACK for input source (microphone into baresip)
                    baresip_bot.do_command("/ausrc jack,MindRootSIP.*")
                    # Use JACK for output playback so baresip exposes decoded audio to JACK.
                    # Pass a specific client name so ports appear as 'MR-STT:output_*'.
                    baresip_bot.do_command("/auplay jack,MR-STT")
                    logger.info("JACK_DEBUG Configured baresip to use JACK (ausrc jack,MindRootSIP.*; auplay jack,MR-STT)")
                else:
                    # Use JACK for audio input (TTS from us to baresip)
                    baresip_bot.do_command("/ausrc jack")
                    # Use JACK for audio output (call audio from baresip for STT)
                    baresip_bot.do_command("/auplay jack")
                    logger.info("JACK_DEBUG Configured baresip to use JACK (ausrc jack; auplay jack)")
            except Exception as e:
                trace = traceback.format_exc()
                logger.error(f"JACK_DEBUG Failed to configure baresip JACK settings: {e}\n{trace}")
    
    def connect_jack_to_baresip(self):
        """Connect JACK ports to baresip after call is established."""
        if self.jack_streamer:
            if self.jack_streamer.connect_to_baresip():
                self.jack_streamer.start_streaming()
                logger.info("JACK audio output ready")
                return True
            else:
                logger.warning("Failed to connect JACK ports")
                return False
        return False
    
    async def send_tts_audio(self, audio_chunk: bytes):
        """Send TTS audio chunk to the SIP call via JACK."""
        if not self.jack_streamer or not self.jack_enabled:
            logger.warning("JACK not available for audio output")
            return
            
        # Unmute JACK if it was muted (from barge-in)
        if self.jack_streamer.muted:
            self.jack_streamer.muted = False
            logger.info("JACK TTS unmuted for new speech")
            
        try:
            # Decode μ-law to 16-bit PCM
            # ElevenLabs sends ulaw_8000 format (8-bit μ-law at 8000 Hz)
            pcm_data = audioop.ulaw2lin(audio_chunk, 2)  # 2 = 16-bit output
            
            # Convert PCM bytes to numpy array
            audio_array = np.frombuffer(pcm_data, dtype=np.int16)
            
            # Convert to float32 normalized to -1.0 to 1.0 for JACK
            audio_float = audio_array.astype(np.float32) / 32768.0
            
            # No resampling needed - audio is already at 8000 Hz from ElevenLabs
            # JACK is also running at 8000 Hz, so we can send directly
            if len(audio_float) > 0:
                # Feed to JACK ring buffer
                self.jack_streamer.write_audio(audio_float)
                
                logger.debug(f"Sent {len(audio_chunk)} bytes TTS audio to JACK")
                
        except Exception as e:
            logger.error(f"Error sending TTS audio: {e}")
    
    def cleanup(self, baresip_bot=None):
        """Cleanup JACK resources."""
        if self.jack_streamer:
            self.jack_streamer.stop_streaming()
            self.jack_streamer.stop()
            
        # Reset baresip audio source
        if baresip_bot:
            baresip_bot.do_command("/ausrc alsa,default")
            
        logger.info("Audio handler cleanup complete")
