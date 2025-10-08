#!/usr/bin/env python3
"""
JACK Audio Streamer for MindRoot SIP Plugin
Adapted from /files/whispertest/jack_streaming_test.py
"""

import jack
import numpy as np
import logging

logger = logging.getLogger(__name__)

class JACKAudioStreamer:
    """JACK client for streaming TTS audio to baresip calls."""
    
    def __init__(self, client_name="MindRootSIP"):
        self.client = jack.Client(client_name)
        self.blocksize = self.client.blocksize
        self.samplerate = self.client.samplerate
        
        logger.info(f"JACK: Blocksize={self.blocksize}, Samplerate={self.samplerate}")
        
        # Create stereo output ports
        self.outports = [
            self.client.outports.register('output_L'),
            self.client.outports.register('output_R')
        ]
        
        # Ring buffer for streaming (20 seconds)
        self.buffer = jack.RingBuffer(self.samplerate * 20 * 4)
        self.streaming = False
        self.muted = False  # Mute flag for barge-in
        
        # Set process callback
        self.client.set_process_callback(self.process)
        
    def process(self, frames):
        """JACK process callback - real-time audio thread."""
        if not self.streaming or self.muted:
            for port in self.outports:
                port.get_array().fill(0)
            return
            
        available = self.buffer.read_space // 4  # 4 bytes per float32
        
        if available >= frames:
            data = self.buffer.read(frames * 4)
            audio = np.frombuffer(data, dtype=np.float32)
            
            # Output to both channels (mono to stereo)
            for port in self.outports:
                port.get_array()[:] = audio
        else:
            # Not enough data, output silence
            for port in self.outports:
                port.get_array().fill(0)
                
    def start(self):
        """Activate JACK client."""
        self.client.activate()
        logger.info(f"JACK client activated: {self.client.name}")
        
    def stop(self):
        """Deactivate JACK client."""
        self.streaming = False
        self.client.deactivate()
        logger.info("JACK client deactivated")
        
    def connect_to_baresip(self) -> bool:
        """Connect output ports to baresip input ports."""
        baresip_ports = self.client.get_ports('baresip', is_audio=True, is_input=True)
        
        if not baresip_ports:
            logger.warning("No baresip input ports found")
            return False
            
        logger.info(f"Found {len(baresip_ports)} baresip input port(s)")
        
        for i, baresip_port in enumerate(baresip_ports[:len(self.outports)]):
            our_port = self.outports[i]
            self.client.connect(our_port, baresip_port)
            logger.info(f"Connected {our_port.name} -> {baresip_port.name}")
            
        return True
        
    def write_audio(self, audio_data: np.ndarray):
        """Write audio data to ring buffer."""
        data_bytes = audio_data.astype(np.float32).tobytes()
        written = self.buffer.write(data_bytes)
        
        if written < len(data_bytes):
            logger.warning(f"Ring buffer full, dropped {len(data_bytes) - written} bytes")
            
    def start_streaming(self):
        """Start streaming audio."""
        self.streaming = True
        logger.info("JACK streaming started")
        
    def stop_streaming(self):
        """Stop streaming audio."""
        self.streaming = False
        logger.info("JACK streaming stopped")
        
    def clear_buffer(self):
        """Clear the ring buffer (for barge-in)."""
        # Read and discard all data in the buffer
        available = self.buffer.read_space
        if available > 0:
            _ = self.buffer.read(available)
            logger.info(f"Cleared {available} bytes from JACK ring buffer")
