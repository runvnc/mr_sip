#!/usr/bin/env python3
"""
Base STT Provider Interface

Defines the abstract interface that all STT providers must implement.
This allows swapping between different STT backends (Whisper, Deepgram, etc.)
without changing the core SIP client code.
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional
from dataclasses import dataclass
import numpy as np

@dataclass
class STTResult:
    """Represents a transcription result from an STT provider."""
    text: str
    is_final: bool
    confidence: float = 1.0
    utterance_num: Optional[int] = None
    timestamp: Optional[float] = None
    
class BaseSTTProvider(ABC):
    """Abstract base class for Speech-to-Text providers."""
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize the STT provider.
        
        Args:
            sample_rate: Audio sample rate in Hz (default: 16000)
        """
        self.sample_rate = sample_rate
        self.is_running = False
        self._on_partial_callback: Optional[Callable[[STTResult], None]] = None
        self._on_final_callback: Optional[Callable[[STTResult], None]] = None
        
    @abstractmethod
    async def start(self) -> None:
        """
        Initialize and start the STT engine.
        Must be called before add_audio().
        """
        pass
        
    @abstractmethod
    async def stop(self) -> None:
        """
        Stop the STT engine and cleanup resources.
        """
        pass
        
    @abstractmethod
    async def add_audio(self, audio_chunk: np.ndarray) -> None:
        """
        Feed audio chunk to the STT engine for processing.
        
        Args:
            audio_chunk: Numpy array of audio samples (float32, mono, normalized to [-1, 1])
        """
        pass
        
    def set_callbacks(self,
                      on_partial: Optional[Callable[[STTResult], None]] = None,
                      on_final: Optional[Callable[[STTResult], None]] = None) -> None:
        """
        Set callbacks for transcription results.
        
        Args:
            on_partial: Called when partial (interim) transcription is available
            on_final: Called when final transcription is available
        """
        self._on_partial_callback = on_partial
        self._on_final_callback = on_final
        
    def _emit_partial(self, result: STTResult) -> None:
        """Emit a partial transcription result."""
        if self._on_partial_callback:
            try:
                self._on_partial_callback(result)
            except Exception as e:
                import logging
                logging.error(f"Error in partial callback: {e}")
                
    def _emit_final(self, result: STTResult) -> None:
        """Emit a final transcription result."""
        if self._on_final_callback:
            try:
                self._on_final_callback(result)
            except Exception as e:
                import logging
                logging.error(f"Error in final callback: {e}")
                
    @abstractmethod
    def get_stats(self) -> dict:
        """
        Get statistics about the STT provider.
        
        Returns:
            dict: Statistics including latency, accuracy, etc.
        """
        pass
