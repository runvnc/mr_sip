#!/usr/bin/env python3
"""
Simple buffered call recorder for PySIP calls (v2 and s2s).

Buffers all audio in memory and writes a stereo WAV at the end.
No incremental flush - avoids window-clipping issues that caused
holes in the outgoing (AI) channel.

Left channel  = incoming (phone)
Right channel = outgoing (AI)
"""

import asyncio
import audioop
import logging
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SimpleRecorder:
    """Buffer-and-write-at-end recorder.

    Public API matches S2SBufferedRecorder / CallRecorder so callers
    can swap without changes.

    Incoming audio is placed using RTP timestamps (accurate).
    Outgoing audio is placed sequentially, anchored to the call
    timeline on the first outgoing frame.
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

        self.sample_rate = 8000

        # (start_sample, ulaw_bytes)
        self._in_segments: List[Tuple[int, bytes]] = []
        self._out_segments: List[Tuple[int, bytes]] = []

        # Incoming sequential position (fallback when no RTP ts)
        self._in_pos_samples: int = 0
        self._in_base_ts: Optional[int] = None

        # Wall-clock reference set on first incoming frame
        self._call_reference_time: Optional[float] = None

        # Outgoing sequential position; None until first outgoing frame
        self._out_pos_samples: Optional[int] = None

        # Map outgoing (perf_counter) onto the incoming media timeline so the
        # gap between caller-end and AI-start reflects real response latency.
        # Both the pacer timestamp and incoming arrival time are perf_counter
        # in this process, so this is single-clock (no cross-clock drift).
        self._last_in_sample: int = 0
        self._last_in_perf: Optional[float] = None
        # Per-turn latency metrics for the *.latency.json sidecar
        self._turn_latencies: List[dict] = []

        self._is_recording: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start_recording(self) -> None:
        if self._is_recording:
            logger.warning(f"SimpleRecorder already started for call {self.call_id}")
            return
        self._is_recording = True
        logger.info(f"SimpleRecorder started for call {self.call_id}")

    async def stop_recording(self) -> None:
        if not self._is_recording:
            return
        self._is_recording = False
        try:
            await asyncio.to_thread(self._build_wavs)
            await asyncio.to_thread(self._write_latency_report)
        except Exception:
            import traceback
            logger.error(traceback.format_exc())

    # ------------------------------------------------------------------
    # Incoming audio
    # ------------------------------------------------------------------

    def record_incoming(self, audio_data: bytes) -> None:
        """Sequential placement (no RTP timestamp available)."""
        if not self._is_recording:
            return
        if self._call_reference_time is None:
            self._call_reference_time = time.perf_counter()
        start = self._in_pos_samples
        self._in_segments.append((start, audio_data))
        self._in_pos_samples += len(audio_data)
        self._last_in_sample = self._in_pos_samples
        self._last_in_perf = time.perf_counter()

    def record_incoming_with_timestamp(self, audio_data: bytes, rtp_timestamp: int) -> None:
        """RTP-timestamp-based placement."""
        if not self._is_recording:
            return
        if self._in_base_ts is None:
            self._in_base_ts = rtp_timestamp
            self._call_reference_time = time.perf_counter()
            logger.info(
                f"SimpleRecorder: call reference time set "
                f"(RTP base={self._in_base_ts})"
            )
        rel_ticks = max(0, rtp_timestamp - self._in_base_ts)
        self._in_segments.append((rel_ticks, audio_data))
        self._last_in_sample = rel_ticks + len(audio_data)
        self._last_in_perf = time.perf_counter()

    # ------------------------------------------------------------------
    # Outgoing audio
    # ------------------------------------------------------------------

    def record_outgoing(self, audio_data: bytes, timestamp: Optional[float] = None) -> None:
        """Place outgoing (AI) audio on the incoming media timeline.

        Both the AudioPacer timestamp and the incoming-frame arrival time are
        perf_counter values from this process, so we map each outgoing frame
        onto the same timeline the caller audio lives on:

            pos = last_incoming_sample + (pacer_ts - last_incoming_arrival) * rate

        This preserves the real silence gap (= response latency) between the
        caller's last audio and the AI's first audio: a 500ms response shows up
        as ~500ms of silence, a 1.2s response as ~1.2s. We only floor against
        true overlap (a negative gap, e.g. eager-stamped TTS); positive gaps
        are never compressed. Placement affects only the offline WAV - the live
        audio path is untouched.
        """
        if not self._is_recording:
            return

        if timestamp is not None and self._last_in_perf is not None:
            # Preferred: single-clock mapping onto the incoming timeline.
            pos = int(self._last_in_sample
                      + (timestamp - self._last_in_perf) * self.sample_rate)
            # Safety floor: never start before the caller's most recent audio.
            if pos < self._last_in_sample:
                pos = self._last_in_sample
        elif timestamp is not None and self._call_reference_time is not None:
            # No incoming anchor yet: fall back to call-relative perf_counter.
            pos = int(max(0.0, timestamp - self._call_reference_time) * self.sample_rate)
        elif self._out_pos_samples is not None:
            # No timestamp: continue sequentially from last outgoing position.
            pos = self._out_pos_samples
        else:
            pos = 0

        # Strictly monotonic within a turn (jitter must not move a frame back);
        # across turns pos jumps forward, which is the real inter-turn gap.
        if self._out_pos_samples is not None and pos < self._out_pos_samples:
            pos = self._out_pos_samples

        # New-turn detection (a forward jump = a gap) -> record a latency metric.
        NEW_TURN_GAP = int(0.12 * self.sample_rate)
        if self._out_pos_samples is None or pos > self._out_pos_samples + NEW_TURN_GAP:
            incoming_end = self._last_in_sample
            self._turn_latencies.append({
                "incoming_end_ms": incoming_end / self.sample_rate * 1000.0,
                "outgoing_start_ms": pos / self.sample_rate * 1000.0,
                "gap_ms": (pos - incoming_end) / self.sample_rate * 1000.0,
            })

        self._out_segments.append((pos, audio_data))
        self._out_pos_samples = pos + len(audio_data)

    # ------------------------------------------------------------------
    # Compatibility no-ops
    # ------------------------------------------------------------------

    def interrupt_outgoing(self) -> None:
        pass

    def interrupt_incoming(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Build WAV files at end of call
    # ------------------------------------------------------------------

    def _build_wavs(self) -> None:
        total_in = max(
            (start + len(buf) for start, buf in self._in_segments), default=0
        )
        total_out = max(
            (start + len(buf) for start, buf in self._out_segments), default=0
        )
        total_samples = max(total_in, total_out)

        if total_samples <= 0:
            logger.info(f"SimpleRecorder {self.call_id}: no audio, skipping")
            return

        if self.record_combined:
            self._write_combined(total_samples)

        if self.record_separate:
            self._write_separate()

    def _write_combined(self, total_samples: int) -> None:
        left = np.zeros(total_samples, dtype=np.int16)
        right = np.zeros(total_samples, dtype=np.int16)

        for start, ulaw_bytes in self._in_segments:
            pcm = np.frombuffer(audioop.ulaw2lin(ulaw_bytes, 2), dtype=np.int16)
            end = min(start + len(ulaw_bytes), total_samples)
            if start < total_samples:
                left[start:end] = pcm[: end - start]

        for start, ulaw_bytes in self._out_segments:
            pcm = np.frombuffer(audioop.ulaw2lin(ulaw_bytes, 2), dtype=np.int16)
            end = min(start + len(ulaw_bytes), total_samples)
            if start < total_samples:
                right[start:end] = pcm[: end - start]

        stereo = np.empty(total_samples * 2, dtype=np.int16)
        stereo[0::2] = left
        stereo[1::2] = right

        with wave.open(str(self.combined_path), "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(self.sample_rate)
            w.writeframes(stereo.tobytes())

        logger.info(
            f"SimpleRecorder: combined WAV saved to {self.combined_path} "
            f"({total_samples} samples)"
        )

    def _write_separate(self) -> None:
        if self._in_segments:
            with wave.open(str(self.incoming_path), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(self.sample_rate)
                for _, ulaw_bytes in self._in_segments:
                    w.writeframes(audioop.ulaw2lin(ulaw_bytes, 2))

        if self._out_segments:
            with wave.open(str(self.outgoing_path), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(self.sample_rate)
                for _, ulaw_bytes in self._out_segments:
                    w.writeframes(audioop.ulaw2lin(ulaw_bytes, 2))

        logger.info(f"SimpleRecorder: separate WAVs written for {self.call_id}")

    def _write_latency_report(self) -> None:
        """Write a *.latency.json sidecar with rough per-turn response gaps.

        gap_ms is the silence between the caller's most recent audio and the
        AI's first audio for that turn (i.e. ballpark response latency). These
        are approximate (pacer/arrival perf_counter based) but good enough to
        judge snappy vs sluggish by eye alongside listening.
        """
        try:
            import json
            if not self._turn_latencies:
                return
            gaps = [t["gap_ms"] for t in self._turn_latencies]
            s = sorted(gaps)
            n = len(s)
            median = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0
            summary = {
                "call_id": self.call_id,
                "turns": n,
                "gap_ms_min": min(gaps),
                "gap_ms_median": median,
                "gap_ms_max": max(gaps),
                "per_turn": self._turn_latencies,
            }
            path = self.combined_path.with_suffix(".latency.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            logger.info(
                "SimpleRecorder latency: %d turns, gap min/median/max = "
                "%.0f/%.0f/%.0f ms -> %s",
                n, min(gaps), median, max(gaps), path,
            )
        except Exception as e:
            logger.error(f"Error writing latency report: {e}")
