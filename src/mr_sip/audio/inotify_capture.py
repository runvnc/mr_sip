#!/usr/bin/env python3
"""
Inotify-based Audio Capture

Monitors baresip's audio recording file using Linux inotify for immediate
notification of new audio data, eliminating polling delays.

This is a drop-in replacement for the file polling approach with much lower
latency and CPU usage.
"""

import os
import asyncio
import logging
import struct
import numpy as np
from pathlib import Path
from typing import Optional, Callable
from scipy import signal

try:
    import inotify.adapters
    INOTIFY_AVAILABLE = True
except ImportError:
    INOTIFY_AVAILABLE = False
    logging.warning("inotify not available. Install with: pip install inotify")

logger = logging.getLogger(__name__)

class InotifyAudioCapture:
    """Captures audio from baresip recording file using inotify."""
    
    def __init__(self, 
                 audio_file: str,
                 target_sample_rate: int = 16000,
                 chunk_callback: Optional[Callable] = None):
        """
        Initialize inotify-based audio capture.
        
        Args:
            audio_file: Path to the baresip audio recording file (WAV format)
            target_sample_rate: Target sample rate for output audio (default: 16000 Hz)
            chunk_callback: Async callback function called with each audio chunk
                           Signature: async callback(audio_chunk: np.ndarray)
        """
        if not INOTIFY_AVAILABLE:
            raise ImportError("inotify module required. Install with: pip install inotify")
            
        self.audio_file = audio_file
        self.target_sample_rate = target_sample_rate
        self.chunk_callback = chunk_callback
        
        # File state
        self.file_handle: Optional[object] = None
        self.data_offset = 0
        self.source_sample_rate = None
        self.channels = None
        self.sample_width = None
        
        # Processing state
        self.is_running = False
        self.monitor_task: Optional[asyncio.Task] = None
        self.inotify: Optional[object] = None
        
        # Stats
        self.total_bytes_read = 0
        self.total_chunks_processed = 0
        
    async def start(self) -> None:
        """Start monitoring the audio file."""
        if self.is_running:
            logger.warning("InotifyAudioCapture already running")
            return
            
        # Wait for file to exist and have data
        await self._wait_for_file()
        
        # Parse WAV header
        if not self._parse_wav_header():
            raise RuntimeError(f"Failed to parse WAV header from {self.audio_file}")
            
        logger.info(f"Audio file format: {self.channels} channels, {self.source_sample_rate} Hz, {self.sample_width} bytes/sample")
        
        # Open file for reading
        self.file_handle = open(self.audio_file, 'rb')
        self.file_handle.seek(self.data_offset)
        
        # Setup inotify
        self.inotify = inotify.adapters.Inotify()
        self.inotify.add_watch(self.audio_file)
        
        self.is_running = True
        
        # Start monitoring task
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        
        logger.info(f"Started inotify monitoring of {self.audio_file}")
        
    async def stop(self) -> None:
        """Stop monitoring the audio file."""
        if not self.is_running:
            return
            
        self.is_running = False
        
        # Cancel monitor task
        if self.monitor_task and not self.monitor_task.done():
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
                
        # Cleanup inotify
        if self.inotify:
            self.inotify.remove_watch(self.audio_file)
            
        # Close file
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None
            
        logger.info(f"Stopped inotify monitoring (processed {self.total_chunks_processed} chunks, {self.total_bytes_read} bytes)")
        
    async def _wait_for_file(self, timeout: float = 5.0) -> None:
        """Wait for audio file to exist and have data."""
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < timeout:
            if os.path.exists(self.audio_file):
                file_size = os.path.getsize(self.audio_file)
                if file_size > 100:  # Has header + some data
                    logger.info(f"Audio file ready: {self.audio_file} ({file_size} bytes)")
                    return
            await asyncio.sleep(0.1)
            
        raise TimeoutError(f"Timeout waiting for audio file: {self.audio_file}")
        
    def _parse_wav_header(self) -> bool:
        """Parse WAV header to get audio format info."""
        try:
            with open(self.audio_file, 'rb') as f:
                # Read RIFF header
                riff = f.read(4)
                if riff != b'RIFF':
                    return False
                    
                file_size = struct.unpack('<I', f.read(4))[0]
                wave_tag = f.read(4)
                if wave_tag != b'WAVE':
                    return False
                
                # Find fmt chunk
                while True:
                    chunk_id = f.read(4)
                    if not chunk_id:
                        return False
                    chunk_size = struct.unpack('<I', f.read(4))[0]
                    
                    if chunk_id == b'fmt ':
                        fmt_data = f.read(chunk_size)
                        self.channels = struct.unpack('<H', fmt_data[2:4])[0]
                        self.source_sample_rate = struct.unpack('<I', fmt_data[4:8])[0]
                        bits_per_sample = struct.unpack('<H', fmt_data[14:16])[0]
                        self.sample_width = bits_per_sample // 8
                        break
                    else:
                        f.seek(chunk_size, 1)
                
                # Find data chunk
                while True:
                    chunk_id = f.read(4)
                    if not chunk_id:
                        return False
                    chunk_size = struct.unpack('<I', f.read(4))[0]
                    
                    if chunk_id == b'data':
                        self.data_offset = f.tell()
                        return True
                    else:
                        f.seek(chunk_size, 1)
                        
        except Exception as e:
            logger.error(f"Error parsing WAV header: {e}")
            return False
            
    async def _monitor_loop(self) -> None:
        """Monitor file for changes using inotify."""
        try:
            # Process any existing data first
            await self._read_and_process_audio()
            
            # Then monitor for new data
            loop = asyncio.get_event_loop()
            
            while self.is_running:
                # Run inotify event check in thread pool (it's blocking)
                events = await loop.run_in_executor(None, self._get_inotify_events)
                
                if events:
                    # File was modified, read new data
                    await self._read_and_process_audio()
                else:
                    # No events, short sleep
                    await asyncio.sleep(0.01)
                    
        except asyncio.CancelledError:
            logger.debug("Monitor loop cancelled")
        except Exception as e:
            if self.is_running:
                logger.error(f"Error in monitor loop: {e}")
                
    def _get_inotify_events(self) -> list:
        """Get inotify events (blocking call, run in executor)."""
        events = []
        for event in self.inotify.event_gen(timeout_s=0.1, yield_nones=False):
            events.append(event)
        return events
        
    async def _read_and_process_audio(self) -> None:
        """Read available audio data and process it."""
        if not self.file_handle:
            return
            
        try:
            # Read available data
            chunk_size = 8000 * self.sample_width * self.channels  # ~0.5 seconds
            audio_data = self.file_handle.read(chunk_size)
            
            if not audio_data:
                return
                
            self.total_bytes_read += len(audio_data)
            
            # Convert to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Convert to float32 in range [-1, 1]
            audio_float = audio_array.astype(np.float32) / 32768.0
            
            # Resample if needed
            if self.source_sample_rate != self.target_sample_rate:
                resample_ratio = self.target_sample_rate / self.source_sample_rate
                num_samples = int(len(audio_float) * resample_ratio)
                audio_float = signal.resample(audio_float, num_samples)
                
            # Call callback
            if self.chunk_callback:
                if asyncio.iscoroutinefunction(self.chunk_callback):
                    await self.chunk_callback(audio_float)
                else:
                    # Run sync callback in thread pool
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, self.chunk_callback, audio_float)
                    
            self.total_chunks_processed += 1
            
        except Exception as e:
            logger.error(f"Error reading/processing audio: {e}")
            
    def get_stats(self) -> dict:
        """Get capture statistics."""
        return {
            "is_running": self.is_running,
            "audio_file": self.audio_file,
            "source_sample_rate": self.source_sample_rate,
            "target_sample_rate": self.target_sample_rate,
            "total_bytes_read": self.total_bytes_read,
            "total_chunks_processed": self.total_chunks_processed
        }
