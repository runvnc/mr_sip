#!/usr/bin/env python3
"""
JACK-based Audio Capture for STT (input path only)

This captures decoded audio directly from baresip's JACK output port, avoiding
file I/O and inotify. It is designed to coexist with the existing JACK output
path used for TTS; it does not modify or depend on that code.

Usage: Create JACKAudioCapture with a chunk_callback and start()/stop().

Notes:
- Uses python-jack-client (import jack)
- Runs a JACK process callback to push float32 audio into a ring buffer
- An asyncio task drains the ring buffer, resamples if needed, and invokes the
  provided async/sync callback with float32 numpy arrays in [-1, 1]
"""

import asyncio
import time
import logging
from typing import Optional, Callable

import numpy as np
from scipy.signal import resample_poly

try:
    import jack
    JACK_AVAILABLE = True
    _jack_import_error = None
except Exception as e:
    JACK_AVAILABLE = False
    _jack_import_error = e

logger = logging.getLogger(__name__)


class JACKAudioCapture:
    """Direct JACK capture from baresip output port to STT callback."""

    def __init__(self,
                 target_sample_rate: int = 16000,
                 chunk_duration_s: float = 0.25,
                 chunk_callback: Optional[Callable] = None,
                 client_name: str = "MindRootSTTIn"):
        if not JACK_AVAILABLE:
            raise ImportError(f"python-jack-client not available: {_jack_import_error}")

        self.target_sample_rate = int(target_sample_rate)
        self.chunk_duration_s = float(chunk_duration_s)
        self.chunk_callback = chunk_callback
        self.client_name = client_name

        self.client: Optional["jack.Client"] = None
        self.inport = None
        self.rb: Optional["jack.RingBuffer"] = None

        self.server_rate: Optional[int] = None
        self.bytes_per_sample = 4  # JACK uses float32 mono per port

        self._reader_task: Optional[asyncio.Task] = None
        self._running = False
        self.selected_port_name: Optional[str] = None

    async def start(self) -> None:
        if self._running:
            logger.warning("JACKAudioCapture already running")
            return

        # Create JACK client and ports
        self.client = jack.Client(self.client_name)
        self.server_rate = int(self.client.samplerate)
        self.inport = self.client.inports.register('input')

        # Allocate ring buffer for ~5 seconds of audio (mono float32)
        rb_bytes = int(self.server_rate * 5 * self.bytes_per_sample)
        self.rb = jack.RingBuffer(rb_bytes)

        @self.client.set_process_callback
        def _process(frames):
            try:
                buf = self.inport.get_array()
                if buf is None:
                    return
                data = buf.tobytes()
                if self.rb is not None:
                    space = self.rb.write_space
                    if space >= len(data):
                        self.rb.write(data)
                    else:
                        if space > 0:
                            self.rb.write(data[:space])
                        # Drop remainder silently to avoid RT thread blocking/logging
            except Exception:
                # Never raise/log from RT thread
                pass

        # Activate client before connecting
        self.client.activate()

        # Attempt to connect baresip -> our inport with retries (ports may appear after call setup)
        self._connect_from_baresip_output(retries=100, delay_s=0.05)

        # Start async reader
        self._running = True
        self._reader_task = asyncio.create_task(self._reader_loop())
        logger.info(f"JACKAudioCapture started at server_rate={self.server_rate} Hz, target={self.target_sample_rate} Hz")

    async def stop(self) -> None:
        if not self._running:
            return

        self._running = False
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
        self._reader_task = None

        try:
            if self.client:
                try:
                    self.client.deactivate()
                finally:
                    self.client.close()
        finally:
            self.client = None
            self.inport = None
            self.rb = None

        logger.info("JACKAudioCapture stopped")

    def _connect_from_baresip_output(self, retries: int = 1, delay_s: float = 0.0) -> None:
        """Find baresip output ports and connect them to our input port, with optional retries.

        Preference order:
          1) Ports with 'baresip' in the name (case-insensitive)
          2) Ports with likely decoded/playback indicators: 'playback','dec','out','output'
        Excludes our own TTS outputs (ports containing 'mindrootsip').
        """
        def score_port(name: str) -> int:
            n = name.lower()
            s = 0
            if 'mindrootsip' in n:
                s -= 1000
            if 'mr-stt' in n:
                s += 200
            if 'mr-stt' in n:
                s += 200
            if 'baresip' in n:
                s += 100
            if 'playback' in n:
                s += 30
            if 'dec' in n:
                s += 25
            if 'output' in n:
                s += 20
            if 'out' in n:
                s += 10
            return s

        for attempt in range(max(1, retries)):
            try:
                ports = self.client.get_ports(is_audio=True, is_output=True)
                if not ports:
                    raise RuntimeError('No JACK audio output ports visible')

                best = None
                best_score = -10**9
                for p in ports:
                    sc = score_port(p.name)
                    if sc > best_score:
                        best = p
                        best_score = sc

                if best is not None and best_score > -100:
                    try:
                        self.client.connect(best, self.inport)
                        self.selected_port_name = best.name
                        logger.info(f"Connected JACK port {best.name} -> {self.client.name}:{self.inport.name}")
                        return
                    except jack.JackError as e:
                        logger.warning(f"Failed to connect {best.name} -> inport: {e}")
                else:
                    logger.debug("No suitable JACK output port found yet; will retry")
            except Exception as e:
                logger.debug(f"JACK port discovery error (attempt {attempt+1}): {e}")

            if attempt < retries - 1:
                time.sleep(max(0.0, delay_s))
        logger.warning("Gave up connecting JACK capture: no suitable baresip output ports found")

    async def _reader_loop(self) -> None:
        """Drain the ring buffer and invoke the chunk callback with float32 arrays."""
        bytes_per_frame = self.bytes_per_sample  # mono float32
        chunk_frames_server = max(1, int(self.chunk_duration_s * self.server_rate))
        chunk_bytes_server = chunk_frames_server * bytes_per_frame
        zero_warned = False
        chunk_count = 0

        try:
            while self._running:
                if self.rb is None:
                    await asyncio.sleep(0.01)
                    continue

                available = self.rb.read_space
                if available < chunk_bytes_server:
                    await asyncio.sleep(0.01)
                    continue

                data = self.rb.read(chunk_bytes_server)
                if not data:
                    await asyncio.sleep(0.005)
                    continue

                audio_f32 = np.frombuffer(data, dtype=np.float32)

                if self.server_rate != self.target_sample_rate:
                    from math import gcd
                    up = int(self.target_sample_rate)
                    down = int(self.server_rate)
                    g = gcd(up, down)
                    up //= g
                    down //= g
                    audio_f32 = resample_poly(audio_f32, up, down).astype(np.float32, copy=False)

                # Basic silence check (RMS)
                chunk_count += 1
                if chunk_count <= 10 or (chunk_count % 50 == 0):
                    rms = float(np.sqrt(np.mean(audio_f32**2))) if audio_f32.size else 0.0
                    logger.debug(f"JACKCapture chunk#{chunk_count} rms={rms:.6f} from='{self.selected_port_name}'")
                    if rms < 1e-5 and not zero_warned:
                        logger.warning("JACKCapture: near-zero audio detected; verify auplay has exposed MR-STT ports and connection.")
                        zero_warned = True

                if self.chunk_callback is not None:
                    try:
                        if asyncio.iscoroutinefunction(self.chunk_callback):
                            await self.chunk_callback(audio_f32)
                        else:
                            loop = asyncio.get_event_loop()
                            await loop.run_in_executor(None, self.chunk_callback, audio_f32)
                    except Exception as cb_e:
                        logger.error(f"Error in JACKAudioCapture callback: {cb_e}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            if self._running:
                logger.error(f"JACKAudioCapture reader error: {e}")
        finally:
            pass

    def get_stats(self) -> dict:
        return {
            "running": self._running,
            "server_rate": self.server_rate,
            "target_rate": self.target_sample_rate,
            "client": self.client_name,
        }
