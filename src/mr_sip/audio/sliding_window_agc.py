#!/usr/bin/env python3
"""
Sliding Window AGC (Automatic Gain Control)

Provides smooth, artifact-free automatic gain control using a sliding window
for RMS calculation and exponential smoothing for gain changes.

This eliminates the pumping/breathing artifacts that occur when AGC is applied
independently to discrete chunks.
"""

import numpy as np
from collections import deque
import logging

logger = logging.getLogger(__name__)

# ANSI color codes for white text on red background
RED_BG_WHITE_TEXT = '\033[41m\033[97m'
RESET_COLOR = '\033[0m'


class SlidingWindowAGC:
    """Smooth AGC using sliding window RMS and exponential gain smoothing."""
    
    def __init__(self,
                 target_rms: float = 0.15,
                 max_gain: float = 4.0,
                 min_gain: float = 0.5,
                 window_seconds: float = 1.5,
                 smoothing: float = 0.95,
                 sample_rate: int = 16000):
        """
        Initialize sliding window AGC.
        
        Args:
            target_rms: Target RMS level (0.15 = good speech level)
            max_gain: Maximum gain multiplier (4.0 = 12dB boost)
            min_gain: Minimum gain multiplier (0.5 = -6dB)
            window_seconds: Window size for RMS calculation (1.5s recommended)
            smoothing: Exponential smoothing factor (0.95 = very smooth, 0.0 = no smoothing)
            sample_rate: Audio sample rate in Hz
        """
        self.target_rms = float(target_rms)
        self.max_gain = float(max_gain)
        self.min_gain = float(min_gain)
        self.window_seconds = float(window_seconds)
        self.smoothing = float(smoothing)
        self.sample_rate = int(sample_rate)
        
        # Calculate window size in samples
        self.window_size = int(self.window_seconds * self.sample_rate)
        
        # Sliding window buffer for RMS calculation
        self.audio_buffer = deque(maxlen=self.window_size)
        
        # Current smoothed gain
        self.current_gain = 1.0
        
        # Statistics
        self.chunks_processed = 0
        self.total_gain_changes = 0.0
        
        print(f"{RED_BG_WHITE_TEXT}[AGC INIT] SlidingWindowAGC initialized: target_rms={target_rms:.3f}, max_gain={max_gain:.1f}, window={window_seconds:.1f}s, smoothing={smoothing:.2f}{RESET_COLOR}")
        
        logger.info(f"SlidingWindowAGC initialized: target_rms={target_rms:.3f}, "
                   f"max_gain={max_gain:.1f}x, window={window_seconds:.1f}s, "
                   f"smoothing={smoothing:.2f}")
    
    def process_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
        """
        Process an audio chunk with sliding window AGC.
        
        Args:
            audio_chunk: Input audio as float32 numpy array in range [-1, 1]
            
        Returns:
            Processed audio with AGC applied, clipped to [-1, 1]
        """
        if audio_chunk.size == 0:
            return audio_chunk
            
        # Ensure float32
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)
        
        # Add current chunk to sliding window
        self.audio_buffer.extend(audio_chunk)
        
        # Calculate RMS over entire sliding window
        if len(self.audio_buffer) > 0:
            window_array = np.array(self.audio_buffer, dtype=np.float32)
            window_rms = float(np.sqrt(np.mean(window_array ** 2)))
        else:
            window_rms = 0.0
        
        # Calculate desired gain based on window RMS
        if window_rms > 1e-6:  # Avoid division by zero
            desired_gain = self.target_rms / window_rms
            # Clamp to min/max gain
            desired_gain = np.clip(desired_gain, self.min_gain, self.max_gain)
        else:
            # Silence - maintain current gain
            desired_gain = self.current_gain
        
        # Smooth the gain change using exponential moving average
        # Higher smoothing = slower gain changes = fewer artifacts
        old_gain = self.current_gain
        self.current_gain = (self.smoothing * self.current_gain + 
                            (1.0 - self.smoothing) * desired_gain)
        
        # Track statistics
        self.chunks_processed += 1
        gain_change = abs(self.current_gain - old_gain)
        self.total_gain_changes += gain_change
        
        # Log occasionally for debugging
        if self.chunks_processed % 100 == 0:
            avg_change = self.total_gain_changes / self.chunks_processed
            print(f"{RED_BG_WHITE_TEXT}[AGC STATS] chunk#{self.chunks_processed}: gain={self.current_gain:.3f}x, window_rms={window_rms:.4f}, avg_change={avg_change:.4f}{RESET_COLOR}")
            logger.debug(f"AGC stats: current_gain={self.current_gain:.3f}, "
                        f"window_rms={window_rms:.4f}, "
                        f"avg_gain_change={avg_change:.4f}")
        
        
        # Apply smoothed gain to current chunk
        processed = audio_chunk * self.current_gain
        
        # Clip to prevent overflow
        processed = np.clip(processed, -1.0, 1.0)
        
        return processed.astype(np.float32)
    
    def reset(self):
        """Reset AGC state (clear buffer and gain)."""
        self.audio_buffer.clear()
        self.current_gain = 1.0
        self.chunks_processed = 0
        self.total_gain_changes = 0.0
        print(f"{RED_BG_WHITE_TEXT}[AGC RESET] SlidingWindowAGC reset{RESET_COLOR}")
        logger.info("SlidingWindowAGC reset")
    
    def get_stats(self) -> dict:
        """Get AGC statistics."""
        avg_change = (self.total_gain_changes / self.chunks_processed 
                     if self.chunks_processed > 0 else 0.0)
        
        return {
            'current_gain': self.current_gain,
            'target_rms': self.target_rms,
            'max_gain': self.max_gain,
            'window_seconds': self.window_seconds,
            'smoothing': self.smoothing,
            'chunks_processed': self.chunks_processed,
            'avg_gain_change_per_chunk': avg_change,
            'buffer_fill': len(self.audio_buffer) / self.window_size
        }
