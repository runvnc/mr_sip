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
from typing import Optional, List, Tuple
import audioop
from collections import deque
import numpy as np
import threading
import array

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
        self.combined_path = self.output_dir / f"{call_id}.wav"
        
        # Frame counters for debugging
        self._incoming_count = 0
        self._outgoing_count = 0
        # Last-frame holders for ticked writer
        self._last_incoming_frame = b'\xff' * 160  # ulaw silence 20ms
        self._last_outgoing_frame = b'\xff' * 160  # ulaw silence 20ms
        self._ticked = True  # enable ticked stereo writer to smooth bursty producers
        self._tick_task: Optional[asyncio.Task] = None
        # Mute window control on interruption/stop (seconds since monotonic)
        self._mute_outgoing_until: float = 0.0
        self._mute_incoming_until: float = 0.0
        
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
                got_outgoing = False

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
                    if len(inc) >= FRAME:
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
                    if len(out) >= FRAME:
                        self._last_outgoing_frame = out[:FRAME]
                        got_outgoing = True
                        if self.record_separate and outgoing_wav:
                            try:
                                outgoing_wav.writeframes(audioop.ulaw2lin(out, 2))
                                frames_written_outgoing += 1
                            except Exception:
                                pass

                if self.record_combined and combined_wav:
                    # Convert both channels (ulaw->PCM16)
                    if time.monotonic() < self._mute_incoming_until:
                        inc_pcm = b'\x00' * (FRAME * 2)
                    else:
                        inc_pcm = audioop.ulaw2lin(self._last_incoming_frame, 2)

                    # During mute window, force PCM-zero silence on outgoing to avoid buzz/DC.
                    # Otherwise, if we did not get a fresh outgoing frame this tick, treat as silence
                    # instead of repeating the last non-silent frame indefinitely.
                    if time.monotonic() < self._mute_outgoing_until or not got_outgoing:
                        out_pcm = b'\x00' * (FRAME * 2)
                    else:
                        out_pcm = audioop.ulaw2lin(self._last_outgoing_frame, 2)

                    # Interleave
                    stereo = b''.join(
                        inc_pcm[i * 2 : i * 2 + 2] + out_pcm[i * 2 : i * 2 + 2]
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
            self._last_incoming_frame = b'\xff' * FRAME_SIZE
            now = time.monotonic()
            self._mute_outgoing_until = now + 0.3
            self._mute_incoming_until = now + 0.3
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


class S2SBufferedRecorder:
    """Buffered, timestamp-aware recorder for S2S (PySIP) calls.

    Instead of writing in (pseudo) real time, this recorder:
    - Buffers incoming (phone) and outgoing (AI) ulaw frames in memory
    - Uses timestamps for outgoing frames to place them on a sample timeline
    - Fills gaps with true PCM-zero silence
    - Builds the final WAV(s) offline in stop_recording()

    Public API is intentionally similar to CallRecorder so MindRootSIPBotS2S
    can swap implementations without large changes.
    """

    def __init__(
        self,
        call_id: str,
        output_dir: str = "recordings",
        record_separate: bool = False,
        record_combined: bool = True,
    ) -> None:
        self.call_id = call_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.record_separate = record_separate
        self.record_combined = record_combined

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.incoming_path = self.output_dir / f"{call_id}_{timestamp}_incoming.wav"
        self.outgoing_path = self.output_dir / f"{call_id}_{timestamp}_outgoing.wav"
        self.combined_path = self.output_dir / f"{call_id}.wav"

        # Sample rate is fixed at 8kHz for ulaw telephony
        self.sample_rate = 8000

        # Segments are (start_sample, ulaw_bytes)
        self._in_segments: List[Tuple[int, bytes]] = []
        self._out_segments: List[Tuple[int, bytes]] = []

        # Track what's been flushed to disk
        self._flushed_in_count: int = 0
        self._flushed_out_count: int = 0
        
        # File handles for incremental writing (opened on first flush)
        self._combined_file: Optional[any] = None
        self._flush_task: Optional[asyncio.Task] = None

        # Incoming: support both sequential and RTP-timestamp-based placement
        self._in_pos_samples: int = 0
        self._in_base_ts: Optional[int] = None

        # Single reference time for the entire call (perf_counter when first incoming frame arrives)
        self._call_reference_time: Optional[float] = None
        
        # Outgoing: sequential placement (timestamps kept for reference/logging only)
        self._out_pos_samples: Optional[int] = None  # None until first outgoing frame
        # Per-response shift applied to clamp an outgoing turn so it never
        # starts before the caller's most recent incoming audio. Recomputed at
        # the start of each outgoing turn; constant within a turn so intra-turn
        # timing is preserved.
        self._out_response_shift: int = 0
        # Ballpark per-turn latency metrics (for judging performance by ear/eye)
        self._turn_latencies: List[dict] = []

        self._is_recording: bool = False

    async def start_recording(self) -> None:
        """Mark recording as started.

        No background tasks are spawned; all work happens in stop_recording().
        """
        if self._is_recording:
            logger.warning(f"Buffered recording already started for call {self.call_id}")
            return
        self._is_recording = True

        # Start background flush task
        self._flush_task = asyncio.create_task(self._flush_loop())

    async def stop_recording(self) -> None:
        """Finalize recording by building WAV files from buffered segments."""
        if not self._is_recording:
            return
        self._is_recording = False

        try:
            # Cancel flush task
            if self._flush_task:
                self._flush_task.cancel()
                try:
                    await self._flush_task
                except asyncio.CancelledError:
                    pass
            
            # Final flush of any remaining segments
            await asyncio.to_thread(self._flush_segments)
            
            # Fix WAV header with correct size and close file
            await asyncio.to_thread(self._finalize_wav)
            
            # Build separate files if requested
            if self.record_separate:
                await asyncio.to_thread(self._build_separate_wavs)

            # Write a small latency report alongside the recording (ballpark)
            await asyncio.to_thread(self._write_latency_report)
        except Exception as e:
            import traceback

            logger.error(traceback.format_exc())

    # API-compatible helpers -------------------------------------------------

    def record_incoming(self, audio_data: bytes) -> None:
        """Buffer incoming ulaw audio (phone -> system).

        Each byte is one 8kHz sample in ulaw format, so len(bytes) == samples.
        """
        if not self._is_recording:
            return

        # Default sequential placement when no RTP timestamp is available
        start = self._in_pos_samples
        self._in_segments.append((start, audio_data))
        self._in_pos_samples += len(audio_data)

    def record_incoming_with_timestamp(self, audio_data: bytes, rtp_timestamp: int) -> None:
        """Buffer incoming ulaw audio using RTP timestamp (ticks @ 8 kHz).

        Args:
            audio_data:    ulaw 8kHz bytes for this jitter frame
            rtp_timestamp: RTP timestamp of the first sample in this frame.
        """
        if not self._is_recording:
            return

        # On first incoming frame, establish the call reference time
        if self._in_base_ts is None:
            self._in_base_ts = rtp_timestamp
            import time as time_module
            self._call_reference_time = time_module.perf_counter()
            logger.info(f"Call reference time established: {self._call_reference_time:.3f} (RTP base: {self._in_base_ts})")

        # Convert RTP timestamp to samples relative to base
        rel_ticks = max(0, rtp_timestamp - self._in_base_ts)
        start_sample = rel_ticks
        
        self._in_segments.append((start_sample, audio_data))

    def _incoming_end_samples(self) -> int:
        """Return the sample position just past the most recently recorded
        incoming (caller) audio. Used to clamp outgoing turns so AI speech is
        never placed before the caller's audio on the shared call timeline.
        Incoming and outgoing share the same zero point (first incoming frame /
        call_reference_time), so positions are directly comparable.
        """
        if not self._in_segments:
            return 0
        last_start, last_buf = self._in_segments[-1]
        return last_start + len(last_buf)

    def record_outgoing(self, audio_data: bytes, timestamp: Optional[float] = None) -> None:
        """Buffer outgoing ulaw audio (system -> phone) with optional timestamp.

        Args:
            audio_data: ulaw 8kHz audio bytes (typically 160-byte frames)
            timestamp:  Absolute playback start time for this frame (perf_counter
                        seconds), as provided by the AudioPacer. Used to place
                        audio at the correct position on the call timeline so
                        that silence gaps between response turns are preserved.
        """
        if not self._is_recording:
            return

        # If we have a timestamp and a call reference time, place this frame
        # at its actual position on the call timeline. This preserves real
        # silence gaps between AI response turns, making latency measurable.
        if timestamp is not None and self._call_reference_time is not None:
            rel_s = timestamp - self._call_reference_time
            raw_start = max(0, int(rel_s * self.sample_rate))

            # Detect the start of a new outgoing turn. The pacer re-anchors its
            # clock on resume, so a new turn shows up as a forward jump well
            # beyond where the previous turn left off.
            NEW_TURN_GAP = int(0.12 * self.sample_rate)  # 120ms
            is_new_turn = (
                self._out_pos_samples is None
                or raw_start > self._out_pos_samples + NEW_TURN_GAP
            )

            if is_new_turn:
                # Clamp: an AI turn must not begin before the caller's most
                # recent incoming audio (eager endpointing can stamp the first
                # TTS chunk at/just-before the caller's real end-of-speech).
                incoming_end = self._incoming_end_samples()
                if raw_start < incoming_end:
                    self._out_response_shift = incoming_end - raw_start
                    logger.info(
                        "Outgoing turn clamped: raw_start=%d -> incoming_end=%d "
                        "(shift=%dms)",
                        raw_start, incoming_end,
                        int(self._out_response_shift / self.sample_rate * 1000),
                    )
                else:
                    self._out_response_shift = 0
                # Record a ballpark latency metric for this turn. raw_gap_ms is
                # the unclamped gap (negative => AI started before caller end).
                raw_gap_ms = (raw_start - incoming_end) / self.sample_rate * 1000.0
                self._turn_latencies.append({
                    "incoming_end_ms": incoming_end / self.sample_rate * 1000.0,
                    "outgoing_start_ms": (raw_start + self._out_response_shift)
                    / self.sample_rate * 1000.0,
                    "raw_gap_ms": raw_gap_ms,
                    "clamped": self._out_response_shift > 0,
                })

            start_sample = raw_start + self._out_response_shift
            # Keep strictly monotonic within a turn (no overlap from jitter).
            if self._out_pos_samples is not None and start_sample < self._out_pos_samples:
                start_sample = self._out_pos_samples
            self._out_segments.append((start_sample, audio_data))
            self._out_pos_samples = start_sample + len(audio_data)
            return

        # Fallback: sequential placement (no timestamp or no reference time yet)
        if self._out_pos_samples is None:
            if self._call_reference_time is not None:
                import time as time_module
                rel_s = time_module.perf_counter() - self._call_reference_time
                self._out_pos_samples = int(rel_s * self.sample_rate)
                logger.debug(f"Outgoing anchor (fallback): {rel_s:.3f}s into call = sample {self._out_pos_samples}")
            else:
                self._out_pos_samples = 0

        start_sample = self._out_pos_samples
        self._out_pos_samples += len(audio_data)
        self._out_segments.append((start_sample, audio_data))
        return

    def interrupt_outgoing(self) -> None:  # compatibility no-op
        """Compatibility hook; no special handling needed for buffered mode."""

    def interrupt_incoming(self) -> None:  # compatibility no-op
        """Compatibility hook; no special handling needed for buffered mode."""

    def _write_latency_report(self) -> None:
        """Write a small JSON sidecar with ballpark per-turn response latency.

        For each AI turn we record:
          - incoming_end_ms:  position of the caller's most recent audio when
                              the turn started (shared call timeline)
          - outgoing_start_ms: where the AI turn was actually placed (after clamp)
          - raw_gap_ms:        unclamped gap (negative => AI audio was stamped
                              before the caller's audio ended, i.e. eager EOT /
                              early-stamping; useful as a diagnostic)
          - clamped:           whether the floor was applied

        These are ballpark numbers (queue/pacer-time based, not wire-time), but
        good enough to judge system responsiveness alongside listening.
        """
        try:
            import json
            if not self._turn_latencies:
                return
            raw_gaps = [t["raw_gap_ms"] for t in self._turn_latencies]
            sorted_gaps = sorted(raw_gaps)
            n = len(sorted_gaps)
            median = sorted_gaps[n // 2] if n % 2 else (
                (sorted_gaps[n // 2 - 1] + sorted_gaps[n // 2]) / 2.0
            )
            summary = {
                "call_id": self.call_id,
                "turns": n,
                "raw_gap_ms_min": min(raw_gaps),
                "raw_gap_ms_median": median,
                "raw_gap_ms_max": max(raw_gaps),
                "turns_clamped": sum(1 for t in self._turn_latencies if t["clamped"]),
                "per_turn": self._turn_latencies,
            }
            report_path = self.combined_path.with_suffix(".latency.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            logger.info(
                "Latency report: %d turns, raw gap min/median/max = %.0f/%.0f/%.0f ms, %d clamped -> %s",
                n, min(raw_gaps), median, max(raw_gaps), summary["turns_clamped"], report_path,
            )
        except Exception as e:
            logger.error(f"Error writing latency report: {e}")

    # Incremental flush implementation ----------------------------------------

    async def _flush_loop(self):
        """Background task that flushes buffered segments to disk periodically."""
        try:
            while self._is_recording:
                await asyncio.sleep(3)  # Flush every 3 seconds (smaller batches = less GIL pressure)
                if self._is_recording:  # Check again after sleep
                    await asyncio.to_thread(self._flush_segments)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in flush loop: {e}")

    def _ensure_wav_open(self):
        """Open the combined WAV file if not already open, write header with placeholder size."""
        if self._combined_file is not None:
            return
        
        if self.record_combined:
            self._combined_file = open(str(self.combined_path), 'wb')
            # Write WAV header with placeholder sizes (will fix on each flush)
            # RIFF header
            self._combined_file.write(b'RIFF')
            self._combined_file.write(b'\x00\x00\x00\x00')  # Placeholder for file size - 8
            self._combined_file.write(b'WAVE')
            # fmt chunk
            self._combined_file.write(b'fmt ')
            self._combined_file.write((16).to_bytes(4, 'little'))  # fmt chunk size
            self._combined_file.write((1).to_bytes(2, 'little'))   # PCM format
            self._combined_file.write((2).to_bytes(2, 'little'))   # 2 channels (stereo)
            self._combined_file.write((self.sample_rate).to_bytes(4, 'little'))  # sample rate
            self._combined_file.write((self.sample_rate * 2 * 2).to_bytes(4, 'little'))  # byte rate
            self._combined_file.write((4).to_bytes(2, 'little'))   # block align (2 channels * 2 bytes)
            self._combined_file.write((16).to_bytes(2, 'little'))  # bits per sample
            # data chunk
            self._combined_file.write(b'data')
            self._combined_file.write(b'\x00\x00\x00\x00')  # Placeholder for data size
            self._combined_file.flush()
            self._total_samples_written = 0

    def _flush_segments(self):
        """Flush new segments to disk. Called from thread pool."""
        if not self.record_combined:
            return
        
        # Get new segments since last flush
        new_in = self._in_segments[self._flushed_in_count:]
        new_out = self._out_segments[self._flushed_out_count:]
        
        if not new_in and not new_out:
            return
        
        self._ensure_wav_open()
        
        if self._combined_file is None:
            return
        
        # Find the range of samples we need to write
        all_new_segments = []
        for start, ulaw_bytes in new_in:
            all_new_segments.append(('in', start, ulaw_bytes))
        for start, ulaw_bytes in new_out:
            all_new_segments.append(('out', start, ulaw_bytes))
        
        if not all_new_segments:
            return
        
        # Find max sample position needed
        max_sample = max(start + len(buf) for _, start, buf in all_new_segments)
        
        # Extend our written range if needed
        if max_sample > self._total_samples_written:
            samples_to_write = max_sample - self._total_samples_written
            
            # Use numpy arrays - C-level ops release the GIL, no pure-Python sample loops
            left = np.zeros(samples_to_write, dtype=np.int16)
            right = np.zeros(samples_to_write, dtype=np.int16)
            
            for channel, start, ulaw_bytes in all_new_segments:
                pcm = audioop.ulaw2lin(ulaw_bytes, 2)
                pcm_np = np.frombuffer(pcm, dtype=np.int16)
                frame_count = len(ulaw_bytes)
                target = left if channel == 'in' else right

                # Map segment position into the current write window
                dst_start = start - self._total_samples_written
                dst_end = dst_start + frame_count
                src_start = max(0, -dst_start)
                dst_start = max(0, dst_start)
                dst_end = min(samples_to_write, dst_end)
                src_end = src_start + (dst_end - dst_start)
                if dst_start < dst_end and src_start < src_end:
                    target[dst_start:dst_end] = pcm_np[src_start:src_end]

            # Interleave with numpy - single C-level operation, GIL released
            stereo = np.empty(samples_to_write * 2, dtype=np.int16)
            stereo[0::2] = left
            stereo[1::2] = right

            self._combined_file.write(stereo.tobytes())
            self._combined_file.flush()
            self._total_samples_written = max_sample
        
        # Update flush counts
        self._flushed_in_count = len(self._in_segments)
        self._flushed_out_count = len(self._out_segments)
        
        # Update header with current size so file is always valid
        self._update_wav_header()
        
        logger.debug(f"Flushed segments: {len(new_in)} in, {len(new_out)} out, total samples: {self._total_samples_written}")

    def _update_wav_header(self):
        """Update WAV header with current sizes (called on each flush)."""
        if self._combined_file is None:
            return
        
        try:
            # Calculate sizes
            data_size = self._total_samples_written * 4  # stereo 16-bit = 4 bytes per sample
            file_size = data_size + 36  # WAV header is 44 bytes, minus 8 for RIFF header = 36
            
            # Remember current position
            current_pos = self._combined_file.tell()
            
            # Seek to RIFF size field (offset 4) and write
            self._combined_file.seek(4)
            self._combined_file.write(file_size.to_bytes(4, 'little'))
            
            # Seek to data size field (offset 40) and write
            self._combined_file.seek(40)
            self._combined_file.write(data_size.to_bytes(4, 'little'))
            
            # Seek back to end for next append
            self._combined_file.seek(current_pos)
            self._combined_file.flush()
        except Exception as e:
            logger.error(f"Error updating WAV header: {e}")

    def _finalize_wav(self):
        """Final header update and close file."""
        if self._combined_file is None:
            return
        
        try:
            # Final header update
            self._update_wav_header()
            
            self._combined_file.close()
            self._combined_file = None
            
            logger.info(
                f"Buffered S2S combined recording saved to {self.combined_path} "
                f"(samples={self._total_samples_written})"
            )
        except Exception as e:
            logger.error(f"Error finalizing WAV: {e}")
            if self._combined_file:
                self._combined_file.close()
                self._combined_file = None

    # Internal helpers --------------------------------------------------------

    def _build_wavs(self) -> None:
        """Construct WAV files from buffered segments."""
        # Determine total length in samples for each side
        total_in = (
            max((start + len(buf) for start, buf in self._in_segments), default=0)
        )
        total_out = (
            max((start + len(buf) for start, buf in self._out_segments), default=0)
        )
        total_samples = max(total_in, total_out)

        if total_samples <= 0:
            logger.info(
                f"Buffered S2S recorder for call {self.call_id}: no audio captured, skipping files"
            )
            return

        # Build combined stereo if requested
        if self.record_combined:
            self._build_combined_wav(total_samples)

        # Optionally build separate mono files; these are best-effort and may
        # ignore exact timing gaps (the combined stereo is the canonical record).
        if self.record_separate:
            self._build_separate_wavs()

    def _build_combined_wav(self, total_samples: int) -> None:
        """Build a stereo WAV (left=incoming, right=outgoing)."""
        left = np.zeros(total_samples, dtype=np.int16)
        right = np.zeros(total_samples, dtype=np.int16)

        # Fill left channel from incoming segments
        for start, ulaw_bytes in self._in_segments:
            pcm = audioop.ulaw2lin(ulaw_bytes, 2)  # 2 bytes/sample
            pcm_np = np.frombuffer(pcm, dtype=np.int16)
            end = min(start + len(ulaw_bytes), total_samples)
            if start < total_samples:
                left[start:end] = pcm_np[:end - start]

        # Fill right channel from outgoing segments using timestamps
        for start, ulaw_bytes in self._out_segments:
            pcm = audioop.ulaw2lin(ulaw_bytes, 2)
            pcm_np = np.frombuffer(pcm, dtype=np.int16)
            end = min(start + len(ulaw_bytes), total_samples)
            if start < total_samples:
                right[start:end] = pcm_np[:end - start]

        with wave.open(str(self.combined_path), "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(self.sample_rate)

            stereo = np.empty(total_samples * 2, dtype=np.int16)
            stereo[0::2] = left
            stereo[1::2] = right
            w.writeframes(stereo.tobytes())

        logger.info(
            f"Buffered S2S combined recording saved to {self.combined_path} "
            f"(samples={total_samples})"
        )

    def _build_separate_wavs(self) -> None:
        """Build simple mono incoming/outgoing WAVs.

        These are written sequentially in segment order and may not reflect
        exact timing gaps; the combined stereo file is the authoritative
        timing-aware recording.
        """
        # Incoming mono
        if self._in_segments:
            with wave.open(str(self.incoming_path), "wb") as w_in:
                w_in.setnchannels(1)
                w_in.setsampwidth(2)
                w_in.setframerate(self.sample_rate)

                for _start, ulaw_bytes in self._in_segments:
                    pcm = audioop.ulaw2lin(ulaw_bytes, 2)
                    w_in.writeframes(pcm)

        # Outgoing mono
        if self._out_segments:
            with wave.open(str(self.outgoing_path), "wb") as w_out:
                w_out.setnchannels(1)
                w_out.setsampwidth(2)
                w_out.setframerate(self.sample_rate)

                for _start, ulaw_bytes in self._out_segments:
                    pcm = audioop.ulaw2lin(ulaw_bytes, 2)
                    w_out.writeframes(pcm)

        logger.info(
            f"Buffered S2S separate mono recordings written for call {self.call_id}"
        )
