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
import time
import wave
import struct
from pathlib import Path
from datetime import datetime
from typing import Optional
import audioop
from collections import deque

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
        # Last-frame holders for ticked writer
        self._last_incoming_frame = b'\xff' * 160  # ulaw silence 20ms
        self._last_outgoing_frame = b'\xff' * 160  # ulaw silence 20ms
        self._ticked = True  # enable ticked stereo writer to smooth bursty producers
        self._tick_task: Optional[asyncio.Task] = None
        # Mute window control for outgoing channel on interruption (seconds since monotonic)
        self._mute_outgoing_until: float = 0.0
        
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
    
    def interrupt_outgoing(self):
        """
        Called when playback is interrupted; replaces the held outgoing frame with silence
        to avoid sustaining the last sample during ticked writes.
        """
        try:
            # Replace held frame with ulaw silence
            self._last_outgoing_frame = b'\xff' * 160  # ulaw silence (20ms)
            # Engage hard mute window for 300ms to prevent residual buzz
            self._mute_outgoing_until = time.monotonic() + 0.3
            # Drain any queued outgoing frames to prevent voiced leftovers after interrupt
            try:
                while True:
                    item = self.outgoing_queue.get_nowait()
                    if item is None:
                        # keep stop sentinel behavior if present
                        break
            except asyncio.QueueEmpty:
                pass
        except Exception:
            pass
            
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
    
    def interrupt_outgoing(self):
        """
        Called when playback is interrupted; replaces the held outgoing frame with silence
        to avoid sustaining the last sample during ticked writes.
        """
        try:
            # Replace held frame with ulaw silence
            self._last_outgoing_frame = b'\xff' * 160  # ulaw silence (20ms)
            # Engage hard mute window for 300ms to prevent residual buzz
            self._mute_outgoing_until = time.monotonic() + 0.3
            # Drain any queued outgoing frames to prevent voiced leftovers after interrupt
            try:
                while True:
                    item = self.outgoing_queue.get_nowait()
                    if item is None:
                        # keep stop sentinel behavior if present
                        break
            except asyncio.QueueEmpty:
                pass
        except Exception:
            pass
            
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
                incoming_wav.setnchannels(1)   # Mono
                incoming_wav.setsampwidth(2)   # 16-bit PCM (converted from ulaw)
                incoming_wav.setframerate(8000)
                # No compression - standard PCM
                
                outgoing_wav = wave.open(str(self.outgoing_path), 'wb')
                # Use PCM16 for consistency across tools
                outgoing_wav.setnchannels(1)
                outgoing_wav.setsampwidth(2)   # 16-bit PCM (converted from ulaw)
                outgoing_wav.setframerate(8000)
                # No compression - standard PCM
            
                
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
            # Helper: write one stereo frame (20ms) using last-known frames or fresh ones if available
            async def write_one_tick():
                nonlocal frames_written_combined, frames_written_incoming, frames_written_outgoing
                FRAME = FRAME_SIZE
                inc = None
                out = None
                # Pull at most one frame per channel per tick; else hold last
                try:
                    inc = self.incoming_queue.get_nowait()
                except asyncio.QueueEmpty:
                    inc = None
                try:
                    out = self.outgoing_queue.get_nowait()
                except asyncio.QueueEmpty:
                    out = None

                # Stop signals: if both queues delivered None and no last frames recorded, end
                if inc is None and out is None and not self._is_recording:
                    return False

                # Update last-frame holders if we got fresh full frames
                if inc is not None:
                    # handle sentinel
                    if inc is None:
                        pass
                    elif len(inc) >= FRAME:
                        # Use only first 160 bytes; if bursts are larger, we consume one frame per tick
                        self._last_incoming_frame = inc[:FRAME]
                        # For separate file, write full available chunk decoded
                        if self.record_separate and incoming_wav:
                            try:
                                incoming_wav.writeframes(audioop.ulaw2lin(inc, 2))
                                frames_written_incoming += 1
                            except Exception:
                                pass
                if out is not None:
                    if out is None:
                        pass
                    elif len(out) >= FRAME:
                        self._last_outgoing_frame = out[:FRAME]
                        if self.record_separate and outgoing_wav:
                            try:
                                outgoing_wav.writeframes(audioop.ulaw2lin(out, 2))
                                frames_written_outgoing += 1
                            except Exception:
                                pass

                if self.record_combined and combined_wav:
                    # Convert both channels (ulaw->PCM16)
                    inc_pcm = audioop.ulaw2lin(self._last_incoming_frame, 2)
                    # During mute window, force PCM-zero silence on outgoing to avoid buzz/DC
                    if time.monotonic() < self._mute_outgoing_until:
                        out_pcm = b'\x00' * (FRAME * 2)
                    else:
                        out_pcm = audioop.ulaw2lin(self._last_outgoing_frame, 2)
                    # Interleave
                    stereo = b''.join(
                        inc_pcm[i*2:i*2+2] + out_pcm[i*2:i*2+2]
                        for i in range(FRAME)
                    )
                    combined_wav.writeframes(stereo)
                    frames_written_combined += 1
                return True

            # Main tick loop at 20ms cadence to smooth bursts from producers (e.g., 250ms OpenAI chunks)
            TICK_SEC = 0.02
            while self._is_recording:
                try:
                    await write_one_tick()
                except Exception as _e:
                    logger.debug(f"Tick write error ignored: {_e}")
                await asyncio.sleep(TICK_SEC)

            # Drain a few final ticks after stop to flush any queued frames,
            # but ensure we don't sustain last outgoing tone after stop
            self._last_outgoing_frame = b'\xff' * FRAME_SIZE
            self._mute_outgoing_until = time.monotonic() + 0.3
            for _ in range(10):
                try:
                    ok = await write_one_tick()
                    if not ok:
                        break
                except Exception:
                    break
            
                        
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
