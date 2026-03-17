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

    # ------------------------------------------------------------------
    # Outgoing audio
    # ------------------------------------------------------------------

    def record_outgoing(self, audio_data: bytes, timestamp: Optional[float] = None) -> None:
        """Sequential placement anchored to call timeline on first frame.

        The timestamp parameter is accepted for API compatibility but
        not used for placement - sequential is cleaner and avoids gaps
        from AudioPacer timing jitter.
        """
        if not self._is_recording:
            return

        if self._out_pos_samples is None:
            # Anchor to current position in the call timeline
            if self._call_reference_time is not None:
                rel_s = time.perf_counter() - self._call_reference_time
                self._out_pos_samples = int(rel_s * self.sample_rate)
                logger.debug(
                    f"SimpleRecorder: outgoing anchor {rel_s:.3f}s "
                    f"= sample {self._out_pos_samples}"
                )
            else:
                self._out_pos_samples = 0

        start = self._out_pos_samples
        self._out_pos_samples += len(audio_data)
        self._out_segments.append((start, audio_data))

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
