#!/usr/bin/env python3
"""
Audio Handler for MindRoot SIP Plugin
Handles audio processing, format conversion, and JACK integration
"""

import numpy as np
import logging
from scipy import signal
from .jack_streamer import JACKAudioStreamer

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
        """Configure baresip to use JACK audio source."""
        if baresip_bot:
            baresip_bot.do_command("/ausrc jack,MindRootSIP.*")
            logger.info("Configured baresip to use JACK")
    
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
            
        try:
            # Convert bytes to numpy array (assuming 16-bit PCM)
            audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
            
            # Convert to float32 normalized to -1.0 to 1.0
            audio_float = audio_array.astype(np.float32) / 32768.0
            
            # Resample if needed (ElevenLabs typically uses 22050 Hz or 44100 Hz)
            # JACK/baresip expects 8000 Hz for telephony
            if len(audio_float) > 0:
                # Assume input is 22050 Hz, resample to 8000 Hz
                target_length = int(len(audio_float) * 8000 / 22050)
                audio_resampled = signal.resample(audio_float, target_length)
                
                # Feed to JACK ring buffer
                self.jack_streamer.write_audio(audio_resampled.astype(np.float32))
                
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
