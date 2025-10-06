#!/usr/bin/env python3
"""
Whisper VAD STT Provider

Wrapper around the existing WhisperStreamingVAD implementation to conform
to the BaseSTTProvider interface. This maintains backward compatibility
while allowing the system to use the new STT provider abstraction.
"""

import logging
import time
import numpy as np
from typing import Optional
from .base_stt import BaseSTTProvider, STTResult
from ..whisper_vad import WhisperStreamingVAD

logger = logging.getLogger(__name__)

class WhisperVADSTT(BaseSTTProvider):
    """Whisper VAD STT provider (existing implementation)."""
    
    def __init__(self,
                 model_size: str = "small",
                 sample_rate: int = 16000,
                 chunk_duration: float = 0.15,
                 silence_threshold: float = 0.005,
                 silence_duration: float = 0.3,
                 min_speech_duration: float = 0.25):
        """
        Initialize Whisper VAD STT provider.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            sample_rate: Audio sample rate in Hz
            chunk_duration: Duration of audio chunks for VAD analysis
            silence_threshold: RMS threshold for silence detection
            silence_duration: Seconds of silence to trigger end of utterance
            min_speech_duration: Minimum speech duration to process
        """
        super().__init__(sample_rate=sample_rate)
        
        self.model_size = model_size
        self.chunk_duration = chunk_duration
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.min_speech_duration = min_speech_duration
        
        self.whisper_vad: Optional[WhisperStreamingVAD] = None
        self.utterance_count = 0
        
    async def start(self) -> None:
        """Initialize and start Whisper VAD."""
        if self.is_running:
            logger.warning("WhisperVADSTT already running")
            return
            
        try:
            logger.info(f"Initializing Whisper VAD (model: {self.model_size})...")
            
            # Create utterance callback that converts to our format
            def utterance_callback(text: str, utterance_num: int, timestamp: float):
                """Callback from WhisperStreamingVAD."""
                result = STTResult(
                    text=text,
                    is_final=True,
                    confidence=1.0,  # Whisper doesn't provide confidence
                    utterance_num=utterance_num,
                    timestamp=timestamp
                )
                self._emit_final(result)
                
            # Initialize WhisperStreamingVAD
            self.whisper_vad = WhisperStreamingVAD(
                model_size=self.model_size,
                sample_rate=self.sample_rate,
                chunk_duration=self.chunk_duration,
                silence_threshold=self.silence_threshold,
                silence_duration=self.silence_duration,
                min_speech_duration=self.min_speech_duration,
                utterance_callback=utterance_callback
            )
            
            # Start the transcriber
            self.whisper_vad.start()
            self.is_running = True
            
            logger.info("Whisper VAD STT provider started")
            
        except Exception as e:
            logger.error(f"Failed to start Whisper VAD: {e}")
            raise
            
    async def stop(self) -> None:
        """Stop Whisper VAD."""
        if not self.is_running:
            return
            
        self.is_running = False
        
        if self.whisper_vad:
            self.whisper_vad.stop()
            
        logger.info("Whisper VAD STT provider stopped")
        
    async def add_audio(self, audio_chunk: np.ndarray) -> None:
        """Feed audio chunk to Whisper VAD."""
        if not self.is_running or not self.whisper_vad:
            logger.warning("Cannot add audio: Whisper VAD not running")
            return
            
        try:
            # WhisperStreamingVAD.add_audio is synchronous
            self.whisper_vad.add_audio(audio_chunk)
        except Exception as e:
            logger.error(f"Error adding audio to Whisper VAD: {e}")
            
    def get_stats(self) -> dict:
        """Get statistics from Whisper VAD."""
        base_stats = {
            "provider": "whisper_vad",
            "model": self.model_size,
            "is_running": self.is_running
        }
        
        if self.whisper_vad:
            whisper_stats = self.whisper_vad.get_stats()
            base_stats.update(whisper_stats)
            
        return base_stats
