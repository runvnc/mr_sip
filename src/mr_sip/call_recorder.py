#!/usr/bin/env python3
"""
Call Recording Module for MindRoot SIP

Provides non-blocking call recording with support for:
- Separate incoming/outgoing streams
- Combined stereo recording (left=incoming, right=outgoing)
- Async I/O to prevent latency impact
- ulaw 8kHz WAV format
"""

import asyncio
import logging
import wave
import struct
from pathlib import Path
from datetime import datetime
from typing import Optional
import audioop

logger = logging.getLogger(__name__)

class CallRecorder:
    """
    Records SIP call audio with minimal latency impact.
    
    Uses async queues and background tasks to offload I/O.
    Supports separate or combined (stereo) recording.
    """
    
    def __init__(self, call_id: str, output_dir: str = "recordings", 
                 record_separate: bool = False, record_combined: bool = True):
        """
        Args:
            call_id: Unique identifier for this call
            output_dir: Directory to save recordings
            record_separate: If True, save separate incoming.wav and outgoing.wav
            record_combined: If True, save combined stereo recording
        """
        self.call_id = call_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.record_separate = record_separate
        self.record_combined = record_combined
        
        # Async queues for audio data (non-blocking) - 5000 frames = ~100 seconds buffer
        self.incoming_queue = asyncio.Queue(maxsize=5000)  # ~100 seconds buffer at 50 fps
        self.outgoing_queue = asyncio.Queue(maxsize=5000)
        
        # Background tasks
        self._recording_task: Optional[asyncio.Task] = None
        self._is_recording = False
        
        # File paths
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.incoming_path = self.output_dir / f"{call_id}_{timestamp}_incoming.wav"
        self.outgoing_path = self.output_dir / f"{call_id}_{timestamp}_outgoing.wav"
        self.combined_path = self.output_dir / f"{call_id}_{timestamp}_combined.wav"
        
        # Frame counters for debugging
        self._incoming_count = 0
        self._outgoing_count = 0
        
    async def start_recording(self):
        """Start recording in background task."""
        if self._is_recording:
            logger.warning(f"Recording already started for call {self.call_id}")
            return
            
        self._is_recording = True
        self._recording_task = asyncio.create_task(self._recording_loop())
        
    async def stop_recording(self):
        """Stop recording and finalize files."""
        if not self._is_recording:
            return
            
        self._is_recording = False
        
        # Signal end of recording
        await self.incoming_queue.put(None)
        await self.outgoing_queue.put(None)
        
        # Wait for recording task to finish
        if self._recording_task:
            try:
                await self._recording_task
            except Exception as e:
                logger.error(f"Error stopping recording: {e}")
                
        
    def record_incoming(self, audio_data: bytes):
        """
        Record incoming audio (phone -> system).
        
        Non-blocking - drops frames if queue is full to prevent latency.
        
        Args:
            audio_data: ulaw 8kHz audio data
        """
        if not self._is_recording:
            return
            
        try:
            self.incoming_queue.put_nowait(audio_data)
            self._incoming_count += 1
        except asyncio.QueueFull:
            # Drop frame to prevent latency - recording is best-effort
            pass  # Silently drop to avoid log spam
            
    def record_outgoing(self, audio_data: bytes):
        """
        Record outgoing audio (system -> phone).
        
        Non-blocking - drops frames if queue is full to prevent latency.
        
        Args:
            audio_data: ulaw 8kHz audio data
        """
        if not self._is_recording:
            return
            
        try:
            self.outgoing_queue.put_nowait(audio_data)
            self._outgoing_count += 1
        except asyncio.QueueFull:
            # Drop frame to prevent latency - recording is best-effort
            pass  # Silently drop to avoid log spam
            
    async def _recording_loop(self):
        """Background task that writes audio to files."""
        try:
            # Unique sentinel to distinguish timeout from stop signal
            _TIMEOUT = object()
            
            # Open WAV files
            incoming_wav = None
            outgoing_wav = None
            combined_wav = None
            
            if self.record_separate:
                incoming_wav = wave.open(str(self.incoming_path), 'wb')
                incoming_wav.setnchannels(1)  # Mono
                incoming_wav.setsampwidth(2)  # 16-bit PCM (converted from ulaw)
                incoming_wav.setframerate(8000)
                # No compression - standard PCM
                
                outgoing_wav = wave.open(str(self.outgoing_path), 'wb')
                outgoing_wav.setnchannels(1)
                outgoing_wav.setsampwidth(1)
                outgoing_wav.setframerate(8000)
                outgoing_wav.setcomptype('ULAW', 'CCITT G.711 u-law')
                
            if self.record_combined:
                combined_wav = wave.open(str(self.combined_path), 'wb')
                combined_wav.setnchannels(2)  # Stereo: left=incoming, right=outgoing
                combined_wav.setsampwidth(2)  # 16-bit PCM (converted from ulaw)
                combined_wav.setframerate(8000)
                # No compression - standard PCM
                
            # Frame counters for combined stereo writes
            frames_written_combined = 0
            
            # Standard frame size for ulaw 8kHz audio (20ms)
            FRAME_SIZE = 160
            
            frames_written_incoming = 0
            frames_written_outgoing = 0
            
            # Track last frame from each channel for padding
            last_incoming = None
            last_outgoing = None
            
            while self._is_recording:
                # Try to get from both queues immediately (no blocking)
                incoming_data = None
                outgoing_data = None
                
                try:
                    incoming_data = self.incoming_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                    
                try:
                    outgoing_data = self.outgoing_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                    
                # Check for end signals
                if incoming_data is None and outgoing_data is None:
                    # Both queues empty, sleep briefly then continue
                    await asyncio.sleep(0.01)  # 10ms sleep
                    continue
                
                # Update last frame tracking
                if incoming_data:
                    last_incoming = incoming_data
                if outgoing_data:
                    last_outgoing = outgoing_data
                
                # Check for stop signals (explicit None from stop_recording)
                if incoming_data is None and last_incoming is None:
                    if outgoing_data is None and last_outgoing is None:
                        break
                    
                # Write separate files
                if self.record_separate:
                    if incoming_data and incoming_wav:
                        # Convert ulaw to PCM before writing
                        pcm_data = audioop.ulaw2lin(incoming_data, 2)  # 2 = 16-bit
                        incoming_wav.writeframes(pcm_data)
                        frames_written_incoming += 1
                    if outgoing_data and outgoing_wav:
                        # Convert ulaw to PCM before writing
                        pcm_data = audioop.ulaw2lin(outgoing_data, 2)  # 2 = 16-bit
                        outgoing_wav.writeframes(pcm_data)
                        frames_written_outgoing += 1
                        
                # Write combined file (stereo interleaved)
                if self.record_combined and combined_wav:
                    # Write frames immediately, padding with silence if one channel is missing
                    # This prevents queue backup and keeps audio in sync
                        
                    # Ensure we have data for both channels (pad with silence if needed)
                    if incoming_data or outgoing_data:
                        # Pad missing channel with silence (ulaw silence = 0xFF)
                        if not incoming_data:
                            incoming_data = b'\xff' * FRAME_SIZE
                        if not outgoing_data:
                            outgoing_data = b'\xff' * FRAME_SIZE
                        
                        # Only write if both are the same size (standard frame)
                        if len(incoming_data) == FRAME_SIZE and len(outgoing_data) == FRAME_SIZE:
                            # Convert both channels from ulaw to PCM
                            incoming_pcm = audioop.ulaw2lin(incoming_data, 2)
                            outgoing_pcm = audioop.ulaw2lin(outgoing_data, 2)
                            
                            # Interleave 16-bit samples: L L R R L L R R ...
                            stereo_data = b''.join(
                                incoming_pcm[i*2:i*2+2] + outgoing_pcm[i*2:i*2+2]
                                for i in range(FRAME_SIZE)
                            )
                            combined_wav.writeframes(stereo_data)
                            frames_written_combined += 1
                        
        except Exception as e:
            logger.error(f"Error in recording loop: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            # Close all files
            if incoming_wav:
                incoming_wav.close()
            if outgoing_wav:
                outgoing_wav.close()
            if combined_wav:
                combined_wav.close()
