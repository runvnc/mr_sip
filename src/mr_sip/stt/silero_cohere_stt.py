"""
Silero VAD + Cohere Transcribe STT Provider

Silero VAD runs locally (CPU-friendly). Cohere Transcribe runs on a remote GPU server.

Audio pipeline:
  ulaw 8kHz (SIP) -> Silero VAD (8kHz native) -> speech detection
  AGC normalization applied per chunk before VAD and speech buffering
  Pre-roll buffer: last ~300ms of audio prepended to speech buffer on speech start
  On speech start  -> fire barge-in callback (turn_resumed)
  Buffer ulaw during speech
  On speech end (eager) -> POST ulaw bytes to COHERE_TRANSCRIBE_URL -> text -> emit eager
  After confirmation delay -> emit final (or cancel if user resumes speaking)

Two-stage end-of-turn detection (similar to Deepgram Flux eager EOT):
  eager_silence_ms  - silence needed for eager EOT (default 500ms)
  final_silence_ms  - total silence needed for final EOT (default 700ms)
  The VAD fires at eager_silence_ms. We transcribe immediately and emit as
  is_eager_eot=True so the agent can start preparing a response. A confirmation
  timer runs for (final_silence_ms - eager_silence_ms). If the user doesn't
  resume speaking, we emit the same text as is_final=True. If the user speaks
  again, we cancel the eager and fire TurnResumed.

Key tuning parameters (all configurable via stt_config dict or env vars):
  threshold              - VAD speech sensitivity (0.0-1.0, default 0.5)
  eager_silence_ms       - silence for eager EOT (default 500ms)
  final_silence_ms       - silence for final EOT (default 700ms)
  speech_pad_ms          - padding added around speech (default 30ms)
  max_utterance_duration_s - hard cap on utterance length (default 30s)
  agc_target_rms         - AGC target RMS level (default 0.1, 0 to disable)
  preroll_ms             - pre-roll buffer size in ms (default 300ms, 0 to disable)
  agc_max_gain           - max gain for per-chunk AGC (default 40x, only if agc_target_rms > 0)
  transcribe_target_rms  - full-buffer normalization before transcription (default 0.15, 0 to disable)

NOTE on AGC: Per-chunk AGC is DISABLED by default (agc_target_rms=0). Google and Silero
both recommend against AGC for speech recognition. Full-buffer normalization is gentler.
"""
import asyncio
import audioop
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from typing import Optional, Callable

import json as _json
import numpy as np
import torch
from collections import deque

try:
    import requests as _requests
except ImportError:
    _requests = None

import urllib.request

from .base_stt import BaseSTTProvider, STTResult

logger = logging.getLogger(__name__)

# VAD chunk size at 8kHz: 256 samples = 256 bytes ulaw = 32ms
VAD_CHUNK_SAMPLES = 256
VAD_SAMPLE_RATE = 8000
COHERE_SAMPLE_RATE = 16000

# Dedicated debug log file - always written regardless of log level
DEBUG_LOG = '/tmp/silero_cohere_stt.log'


def _dlog(msg: str):
    """Write a timestamped line (with ms resolution) to the debug log file and also to logger.info."""
    now = datetime.now()
    ts = now.strftime('%Y-%m-%d %H:%M:%S') + f'.{now.microsecond // 1000:03d}'
    line = f'[{ts}] {msg}'
    try:
        with open(DEBUG_LOG, 'a') as f:
            f.write(line + '\n')
            f.flush()
    except Exception:
        pass
    logger.info(msg)


class SileroCohereSTT(BaseSTTProvider):
    """
    Local VAD + remote ASR STT provider with two-stage eager end-of-turn.

    Barge-in detection: Silero VAD fires on every speech onset.
    The turn_resumed_callback is called immediately, which halts AI audio
    output and cancels any in-progress LLM response.

    Eager EOT: After eager_silence_ms of silence, transcription fires and
    emits is_eager_eot=True. After final_silence_ms total silence, emits
    is_final=True. If user resumes speaking before final, cancels eager.
    """

    def __init__(
        self,
        sample_rate: int = 8000,
        threshold: float = 0.5,
        min_silence_duration_ms: int = 600,
        speech_pad_ms: int = 30,
        language: str = 'en',
        cohere_model_id: str = 'CohereLabs/cohere-transcribe-03-2026',
        device: Optional[str] = None,
        max_utterance_duration_s: float = 30.0,
        cohere_transcribe_url: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate)

        self.threshold = float(os.getenv('SILERO_VAD_THRESHOLD', str(threshold)))

        # Two-stage silence detection
        # eager_silence_ms: VAD fires here, we transcribe and emit eager EOT
        # final_silence_ms: confirmation timer expires, we emit final EOT
        self._eager_silence_ms = int(
            os.getenv('SILERO_EAGER_SILENCE_MS', '500')
        )
        self._final_silence_ms = int(
            os.getenv('SILERO_FINAL_SILENCE_MS', '700')
        )
        # VADIterator uses eager silence as its min_silence_duration_ms
        self.min_silence_duration_ms = self._eager_silence_ms

        self.speech_pad_ms = int(os.getenv('SILERO_SPEECH_PAD_MS', str(speech_pad_ms)))
        self.language = os.getenv('COHERE_TRANSCRIBE_LANGUAGE', language)
        self.cohere_model_id = os.getenv('COHERE_TRANSCRIBE_MODEL', cohere_model_id)
        self.max_utterance_duration_s = float(
            os.getenv('COHERE_MAX_UTTERANCE_S', str(max_utterance_duration_s))
        )

        # Per-chunk AGC: normalize each chunk to this RMS level (0 = disabled)
        # DISABLED by default - per-chunk AGC is too aggressive for speech recognition.
        # It destroys dynamic range, creates pumping artifacts, and amplifies noise.
        # Google explicitly recommends against AGC for ASR.
        self.agc_target_rms = float(
            os.getenv('SILERO_AGC_TARGET_RMS', '0')
        )
        # Max gain cap for per-chunk AGC (only used if agc_target_rms > 0)
        self.agc_max_gain = float(
            os.getenv('SILERO_AGC_MAX_GAIN', '40.0')
        )
        # Full-buffer normalization: gentler than per-chunk AGC, uses whole utterance RMS.
        # Applied once before transcription. More stable since it sees the full utterance.
        self.transcribe_target_rms = float(
            os.getenv('SILERO_TRANSCRIBE_TARGET_RMS', '0.15')
        )

        # Pre-roll: keep a rolling buffer of recent chunks to prepend on speech start
        preroll_ms = int(os.getenv('SILERO_PREROLL_MS', '300'))
        self._preroll_chunks = max(0, preroll_ms // 32)
        self._preroll_buffer: deque = deque(maxlen=self._preroll_chunks) if self._preroll_chunks > 0 else deque(maxlen=0)

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.device = os.getenv('COHERE_TRANSCRIBE_DEVICE', self.device)

        self.cohere_transcribe_url = (
            cohere_transcribe_url
            or os.getenv('COHERE_TRANSCRIBE_URL', '')
        ).rstrip('/')

        # Persistent HTTP session for remote transcription (keep-alive)
        self._http_session = None
        if _requests is not None and self.cohere_transcribe_url:
            self._http_session = _requests.Session()
            self._http_session.headers.update({
                'Content-Type': 'application/octet-stream',
                'User-Agent': 'mr-sip/1.0',
            })

        # Models (only used for local fallback)
        self.vad_model = None
        self.vad_iterator = None
        self.cohere_model = None
        self.cohere_processor = None

        # Audio state
        self._vad_buffer: bytes = b''
        self._speech_buffer: bytes = b''
        self._is_speaking: bool = False
        self._speech_start_time: float = 0.0
        self._last_speech_audio_time: float = 0.0  # perf_counter when last speech audio was buffered
        self._frames_received: int = 0
        self._vad_chunks_processed: int = 0

        # Eager EOT state
        self._eager_pending: bool = False
        self._eager_text: str = ''
        self._eager_timer_task: Optional[asyncio.Task] = None
        self._eager_utterance_num: int = 0

        # Callbacks
        self._turn_resumed_callback: Optional[Callable] = None

        # Stats
        self._utterance_count: int = 0
        self._total_audio_bytes: int = 0
        self._transcription_times: list = []
        self._total_eager_eots: int = 0
        self._total_eager_confirmed: int = 0
        self._total_eager_cancelled: int = 0
        # VAD timing stats (rolling averages)
        self._vad_preprocess_times: list = []  # ulaw decode + AGC per chunk
        self._vad_inference_times: list = []    # VAD model call per chunk
        self._vad_total_times: list = []        # full _process_vad_chunk time

        _dlog(
            f'SileroCohereSTT.__init__: threshold={self.threshold}, '
            f'eager_silence={self._eager_silence_ms}ms, '
            f'final_silence={self._final_silence_ms}ms, '
            f'speech_pad={self.speech_pad_ms}ms, '
            f'agc_target_rms={self.agc_target_rms}, '
            f'agc_max_gain={self.agc_max_gain}x, '
            f'transcribe_target_rms={self.transcribe_target_rms}, '
            f'preroll_chunks={self._preroll_chunks} (~{self._preroll_chunks * 32}ms), '
            f'http_session={"requests" if self._http_session else "urllib"}, '
            f'remote_url="{self.cohere_transcribe_url or "(none - local fallback)"}", '
            f'language={self.language}'
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Load Silero VAD. Validate remote URL if configured."""
        if self.is_running:
            _dlog('SileroCohereSTT.start: already running, skipping')
            return

        _dlog('SileroCohereSTT.start: loading Silero VAD model...')
        try:
            await asyncio.get_event_loop().run_in_executor(None, self._load_vad_model)
        except Exception as e:
            _dlog(f'SileroCohereSTT.start: FATAL - failed to load Silero VAD: {e}')
            _dlog(traceback.format_exc())
            raise RuntimeError(
                f'SileroCohereSTT: failed to load Silero VAD model. '
                f'Is silero-vad installed? Error: {e}'
            ) from e

        _dlog('SileroCohereSTT.start: Silero VAD loaded OK.')

        if self.cohere_transcribe_url:
            _dlog(f'SileroCohereSTT.start: checking remote Cohere Transcribe at {self.cohere_transcribe_url} ...')
            try:
                health_url = f'{self.cohere_transcribe_url}/health'
                req = urllib.request.Request(
                    health_url,
                    headers={'User-Agent': 'mr-sip/1.0'},
                    method='GET',
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    body = _json.loads(resp.read())
                _dlog(f'SileroCohereSTT.start: remote health OK: {body}')
            except Exception as e:
                _dlog(f'SileroCohereSTT.start: FATAL - remote Cohere Transcribe not reachable at {self.cohere_transcribe_url}: {e}')
                raise RuntimeError(
                    f'SileroCohereSTT: remote Cohere Transcribe server not reachable at '
                    f'{self.cohere_transcribe_url}/health. '
                    f'Set COHERE_TRANSCRIBE_URL correctly or start the server. Error: {e}'
                ) from e
        else:
            raise RuntimeError(
                'SileroCohereSTT: COHERE_TRANSCRIBE_URL is not set. '
                'Local Cohere fallback is disabled (too slow). '
                'Set COHERE_TRANSCRIBE_URL to the remote transcription server URL.'
            )

        self.is_running = True
        _dlog('SileroCohereSTT.start: provider started and ready.')

    def _load_vad_model(self):
        """Load Silero VAD (blocking, run in executor)."""
        try:
            from silero_vad import load_silero_vad, VADIterator
        except ImportError:
            raise ImportError(
                'silero-vad package not installed. Run: pip install silero-vad'
            )
        self.vad_model = load_silero_vad(onnx=False)
        # VAD fires at eager_silence_ms; final confirmation is handled by our timer
        self.vad_iterator = VADIterator(
            model=self.vad_model,
            threshold=self.threshold,
            sampling_rate=VAD_SAMPLE_RATE,
            min_silence_duration_ms=self._eager_silence_ms,
            speech_pad_ms=self.speech_pad_ms,
        )
        _dlog(f'_load_vad_model: VADIterator created (threshold={self.threshold}, '
              f'min_silence={self._eager_silence_ms}ms [eager], sr={VAD_SAMPLE_RATE})')

    def _load_cohere_model(self):
        """Load Cohere Transcribe model locally. DISABLED - too slow."""
        raise RuntimeError(
            'Local Cohere model is disabled. Set COHERE_TRANSCRIBE_URL to use remote transcription server.'
        )

    async def stop(self) -> None:
        """Stop the provider and release resources."""
        if not self.is_running:
            return
        self.is_running = False
        self._is_speaking = False
        self._vad_buffer = b''
        self._speech_buffer = b''
        self._preroll_buffer.clear()
        # Cancel any pending eager confirmation
        if self._eager_timer_task and not self._eager_timer_task.done():
            self._eager_timer_task.cancel()
            self._eager_timer_task = None
        self._eager_pending = False
        self._eager_text = ''
        # Close persistent HTTP session
        if self._http_session is not None:
            try:
                self._http_session.close()
            except Exception:
                pass
        if self.vad_iterator is not None:
            try:
                self.vad_iterator.reset_states()
            except Exception:
                pass
        _dlog(f'SileroCohereSTT stopped. Stats: frames={self._frames_received}, '
              f'vad_chunks={self._vad_chunks_processed}, utterances={self._utterance_count}, '
              f'eager_eots={self._total_eager_eots}, confirmed={self._total_eager_confirmed}, '
              f'cancelled={self._total_eager_cancelled}')

    # ------------------------------------------------------------------
    # Audio ingestion
    # ------------------------------------------------------------------

    async def add_audio_bytes(self, audio_bytes: bytes) -> None:
        """
        Feed raw ulaw 8kHz bytes from SIP into the VAD pipeline.
        Called for every RTP frame (~20ms / 160 bytes).
        """
        if not self.is_running:
            return

        self._frames_received += 1
        self._total_audio_bytes += len(audio_bytes)

        # Periodic heartbeat log every 500 frames (~10s)
        if self._frames_received % 500 == 0:
            _dlog(
                f'add_audio_bytes: heartbeat - frames={self._frames_received}, '
                f'vad_chunks={self._vad_chunks_processed}, '
                f'is_speaking={self._is_speaking}, '
                f'speech_buf={len(self._speech_buffer)}B, '
                f'eager_pending={self._eager_pending}, '
                f'utterances={self._utterance_count}'
            )

        # Buffer audio during speech for transcription
        if self._is_speaking:
            self._speech_buffer += audio_bytes
            self._last_speech_audio_time = time.perf_counter()
            max_bytes = int(self.max_utterance_duration_s * VAD_SAMPLE_RATE)
            if len(self._speech_buffer) > max_bytes:
                _dlog(f'add_audio_bytes: utterance exceeded {self.max_utterance_duration_s}s limit, forcing end-of-speech')
                await self._on_speech_end()
                return

        # Accumulate into VAD buffer
        self._vad_buffer += audio_bytes

        # Process complete VAD chunks
        while len(self._vad_buffer) >= VAD_CHUNK_SAMPLES:
            chunk_bytes = self._vad_buffer[:VAD_CHUNK_SAMPLES]
            self._vad_buffer = self._vad_buffer[VAD_CHUNK_SAMPLES:]
            await self._process_vad_chunk(chunk_bytes)

    async def add_audio(self, audio_chunk: np.ndarray) -> None:
        """Compatibility: accept float32 numpy audio and convert to ulaw bytes."""
        if not self.is_running:
            return
        audio_int16 = (np.clip(audio_chunk, -1.0, 1.0) * 32767).astype(np.int16)
        pcm_bytes = audio_int16.tobytes()
        ulaw_bytes = audioop.lin2ulaw(pcm_bytes, 2)
        await self.add_audio_bytes(ulaw_bytes)

    # ------------------------------------------------------------------
    # VAD processing
    # ------------------------------------------------------------------

    async def _process_vad_chunk(self, chunk_bytes: bytes) -> None:
        """Run one 256-byte (32ms) ulaw chunk through Silero VAD."""
        t_chunk_start = time.perf_counter()
        try:
            self._vad_chunks_processed += 1

            # Log first chunk to confirm VAD is receiving audio
            if self._vad_chunks_processed == 1:
                _dlog('_process_vad_chunk: first VAD chunk received - VAD is active')

            t_preprocess = time.perf_counter()

            # ulaw -> PCM int16 -> float32 tensor
            pcm_bytes = audioop.ulaw2lin(chunk_bytes, 2)
            audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
            audio_float = audio_int16.astype(np.float32) / 32768.0

            # AGC: normalize to target RMS to handle low-volume phones
            if self.agc_target_rms > 0:
                rms = float(np.sqrt(np.mean(audio_float ** 2)))
                if rms > 1e-6:
                    gain = self.agc_target_rms / rms
                    gain = min(gain, self.agc_max_gain)
                    audio_float = np.clip(audio_float * gain, -1.0, 1.0)
                    # Re-encode normalized audio back to ulaw for speech buffer
                    norm_int16 = (audio_float * 32767).astype(np.int16)
                    chunk_bytes = audioop.lin2ulaw(norm_int16.tobytes(), 2)

            chunk_tensor = torch.from_numpy(audio_float)

            preprocess_us = (time.perf_counter() - t_preprocess) * 1e6

            # VADIterator returns {'start': N} or {'end': N} or None
            t_vad = time.perf_counter()
            result = self.vad_iterator(chunk_tensor, return_seconds=False)
            vad_us = (time.perf_counter() - t_vad) * 1e6

            chunk_total_us = (time.perf_counter() - t_chunk_start) * 1e6

            if result is None:
                # Always update pre-roll buffer (before speech start)
                if not self._is_speaking:
                    self._preroll_buffer.append(chunk_bytes)
                return

            # Log timing on VAD events and periodically
            self._vad_preprocess_times.append(preprocess_us)
            self._vad_inference_times.append(vad_us)
            self._vad_total_times.append(chunk_total_us)

            _dlog(f'_process_vad_chunk: VAD event: {result} | '
                  f'preprocess={preprocess_us:.0f}us vad={vad_us:.0f}us total={chunk_total_us:.0f}us')

            # Log rolling averages every 50 events
            if len(self._vad_inference_times) % 50 == 0:
                avg_pre = sum(self._vad_preprocess_times[-50:]) / min(50, len(self._vad_preprocess_times))
                avg_vad = sum(self._vad_inference_times[-50:]) / min(50, len(self._vad_inference_times))
                avg_tot = sum(self._vad_total_times[-50:]) / min(50, len(self._vad_total_times))
                _dlog(f'[VAD TIMING] avg over last 50 events: preprocess={avg_pre:.0f}us '
                      f'vad_inference={avg_vad:.0f}us total={avg_tot:.0f}us')
                # Trim to prevent unbounded growth
                if len(self._vad_inference_times) > 200:
                    self._vad_preprocess_times = self._vad_preprocess_times[-100:]
                    self._vad_inference_times = self._vad_inference_times[-100:]
                    self._vad_total_times = self._vad_total_times[-100:]

            # Always update pre-roll buffer with normalized chunk (before speech start)
            if not self._is_speaking:
                self._preroll_buffer.append(chunk_bytes)

            if 'start' in result:
                await self._on_speech_start()
            elif 'end' in result:
                await self._on_speech_end()

        except Exception as e:
            _dlog(f'_process_vad_chunk: ERROR: {e}\n{traceback.format_exc()}')

    async def _on_speech_start(self) -> None:
        """Called when Silero VAD detects speech onset."""
        if self._is_speaking:
            return

        # Cancel any pending eager confirmation - user is speaking again
        if self._eager_pending:
            self._total_eager_cancelled += 1
            _dlog(f'[EAGER] Cancelled eager EOT #{self._eager_utterance_num} - user resumed speaking')
            if self._eager_timer_task and not self._eager_timer_task.done():
                self._eager_timer_task.cancel()
                self._eager_timer_task = None
            self._eager_pending = False
            self._eager_text = ''

        self._is_speaking = True
        self._speech_start_time = time.perf_counter()
        self._last_speech_audio_time = time.perf_counter()
        self._speech_buffer = b''

        # Prepend pre-roll buffer to capture audio before VAD trigger
        if self._preroll_buffer:
            preroll_bytes = b''.join(self._preroll_buffer)
            self._speech_buffer = preroll_bytes
            _dlog(f'[VAD] Pre-roll: prepended {len(preroll_bytes)} bytes ({len(self._preroll_buffer)} chunks)')

        _dlog(f'[VAD] Speech START (utterance #{self._utterance_count + 1})')

        if self._turn_resumed_callback is not None:
            try:
                _dlog('[VAD] Firing turn_resumed_callback (barge-in)')
                self._turn_resumed_callback()
            except Exception as e:
                _dlog(f'_on_speech_start: turn_resumed_callback error: {e}')

    async def _on_speech_end(self) -> None:
        """Called when Silero VAD detects end-of-speech (at eager_silence_ms).

        Transcribes buffered audio and emits as eager EOT.
        Starts a confirmation timer for final EOT.
        """
        if not self._is_speaking:
            return

        self._is_speaking = False
        vad_end_time = time.perf_counter()
        speech_duration = vad_end_time - self._speech_start_time
        time_since_last_audio = (vad_end_time - self._last_speech_audio_time) * 1000
        speech_bytes = self._speech_buffer
        self._speech_buffer = b''

        _dlog(f'[VAD] Speech END (eager): {speech_duration:.2f}s, {len(speech_bytes)} bytes buffered, '
              f'time_since_last_speech_audio={time_since_last_audio:.0f}ms')

        if len(speech_bytes) < VAD_CHUNK_SAMPLES * 2:
            _dlog('[VAD] Speech segment too short (<512 bytes), skipping transcription')
            return

        # Full-buffer normalization
        t_norm = time.perf_counter()
        speech_bytes = self._normalize_buffer(speech_bytes)
        norm_ms = (time.perf_counter() - t_norm) * 1000

        t0 = time.perf_counter()
        _dlog(f'[TRANSCRIBE] Starting transcription of {len(speech_bytes)} bytes (normalize={norm_ms:.1f}ms)...')
        try:
            text = await asyncio.get_event_loop().run_in_executor(
                None, self._transcribe_ulaw, speech_bytes
            )
        except Exception as e:
            _dlog(f'[TRANSCRIBE] ERROR: {e}\n{traceback.format_exc()}')
            return

        elapsed = time.time() - (t0 + (time.time() - time.perf_counter()))  # wall clock approx
        elapsed_pc = (time.perf_counter() - t0)
        self._transcription_times.append(elapsed_pc)
        total_since_vad_end = (time.perf_counter() - vad_end_time) * 1000
        total_since_last_audio = (time.perf_counter() - self._last_speech_audio_time) * 1000
        _dlog(f'[TRANSCRIBE] Done in {elapsed_pc*1000:.0f}ms -> "{text}" | '
              f'total_since_vad_end={total_since_vad_end:.0f}ms | '
              f'total_since_last_speech_audio={total_since_last_audio:.0f}ms')

        if not text or not text.strip():
            _dlog('[TRANSCRIBE] Empty result, skipping emit')
            return

        text = text.strip()
        self._utterance_count += 1
        self._total_eager_eots += 1

        # Emit as eager EOT
        eager_result = STTResult(
            text=text,
            is_final=False,
            is_eager_eot=True,
            confidence=0.8,
            timestamp=time.time(),
        )
        eager_result.utterance_num = self._utterance_count
        _dlog(f'[EMIT] Eager EOT #{self._utterance_count}: "{text}"')
        self._emit_partial(eager_result)

        # Store eager state and start confirmation timer
        self._eager_pending = True
        self._eager_text = text
        self._eager_utterance_num = self._utterance_count

        confirmation_delay_ms = max(0, self._final_silence_ms - self._eager_silence_ms)
        _dlog(f'[EAGER] Starting confirmation timer: {confirmation_delay_ms}ms')
        self._eager_timer_task = asyncio.ensure_future(
            self._eager_confirmation_timer(confirmation_delay_ms / 1000.0)
        )

    async def _eager_confirmation_timer(self, delay_seconds: float) -> None:
        """Wait for confirmation delay, then emit final if still pending."""
        try:
            await asyncio.sleep(delay_seconds)
        except asyncio.CancelledError:
            _dlog('[EAGER] Confirmation timer cancelled')
            return

        if not self._eager_pending:
            _dlog('[EAGER] Confirmation timer fired but eager no longer pending')
            return

        # Confirm: emit as final
        self._eager_pending = False
        self._total_eager_confirmed += 1
        text = self._eager_text
        utterance_num = self._eager_utterance_num
        self._eager_text = ''

        result = STTResult(
            text=text,
            is_final=True,
            is_eager_eot=False,
            confidence=0.95,
            timestamp=time.time(),
        )
        result.utterance_num = utterance_num
        _dlog(f'[EMIT] Final (confirmed) #{utterance_num}: "{text}"')
        self._emit_final(result)

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    def _transcribe_ulaw(self, ulaw_bytes: bytes) -> str:
        """Route to remote or local transcription."""
        if self.cohere_transcribe_url:
            return self._transcribe_remote(ulaw_bytes)
        return self._transcribe_local(ulaw_bytes)

    def _normalize_buffer(self, ulaw_bytes: bytes) -> bytes:
        """Normalize the full speech buffer to a target RMS level.

        Applied once before transcription for stable level normalization.
        More accurate than per-chunk AGC since it uses the full utterance RMS.
        """
        if self.transcribe_target_rms <= 0:
            return ulaw_bytes
        pcm_bytes = audioop.ulaw2lin(ulaw_bytes, 2)
        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(audio_float ** 2)))
        if rms < 1e-6:
            return ulaw_bytes  # Silent buffer, nothing to normalize
        gain = self.transcribe_target_rms / rms
        # Cap at 10x - higher gains just amplify noise and hurt ASR accuracy
        gain = min(gain, 10.0)
        audio_float = np.clip(audio_float * gain, -1.0, 1.0)
        _dlog(f'[AGC] full-buffer normalize: rms={rms:.4f} gain={gain:.1f}x target={self.transcribe_target_rms}')
        norm_int16 = (audio_float * 32767).astype(np.int16)
        return audioop.lin2ulaw(norm_int16.tobytes(), 2)

    def _transcribe_remote(self, ulaw_bytes: bytes) -> str:
        """POST ulaw bytes to the remote Cohere Transcribe HTTP server.

        Uses persistent requests.Session (keep-alive) when available,
        falls back to urllib for single-shot requests.
        """
        url = f'{self.cohere_transcribe_url}/transcribe?language={self.language}'
        _dlog(f'_transcribe_remote: POST {len(ulaw_bytes)} bytes to {url}')

        # Prefer persistent session (requests library)
        if self._http_session is not None:
            try:
                resp = self._http_session.post(
                    url,
                    data=ulaw_bytes,
                    timeout=30,
                )
                body = resp.json()
                text = body.get('text', '')
                _dlog(f'_transcribe_remote: response: {body}')
                return text
            except Exception as e:
                _dlog(f'_transcribe_remote (requests): FAILED: {e}\n{traceback.format_exc()}')
                return ''

        # Fallback: urllib (no keep-alive)
        req = urllib.request.Request(
            url,
            data=ulaw_bytes,
            headers={'Content-Type': 'application/octet-stream',
                     'User-Agent': 'mr-sip/1.0'},
            method='POST',
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = _json.loads(resp.read())
                text = body.get('text', '')
                _dlog(f'_transcribe_remote: response: {body}')
                return text
        except Exception as e:
            _dlog(f'_transcribe_remote: FAILED: {e}\n{traceback.format_exc()}')
            return ''

    def _transcribe_local(self, ulaw_bytes: bytes) -> str:
        """Run Cohere Transcribe locally. DISABLED - too slow."""
        raise RuntimeError(
            'Local Cohere transcription is disabled. Set COHERE_TRANSCRIBE_URL to use remote transcription server.'
        )

    @staticmethod
    def _resample_2x(audio: np.ndarray) -> np.ndarray:
        """Upsample by 2x via linear interpolation (8kHz -> 16kHz)."""
        n = len(audio)
        out = np.empty(n * 2, dtype=np.float32)
        out[0::2] = audio
        out[1::2] = np.concatenate([
            (audio[:-1] + audio[1:]) / 2.0,
            [audio[-1]],
        ])
        return out

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def set_turn_resumed_callback(self, callback: Optional[Callable]) -> None:
        """Set the barge-in callback (called on every speech onset)."""
        self._turn_resumed_callback = callback
        _dlog(f'set_turn_resumed_callback: callback set to {callback}')

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        avg_transcription_ms = (
            sum(self._transcription_times) / len(self._transcription_times) * 1000
            if self._transcription_times else 0
        )
        return {
            'provider': 'silero_cohere',
            'vad_model': 'silero_vad_v6',
            'asr_model': self.cohere_model_id,
            'is_running': self.is_running,
            'utterance_count': self._utterance_count,
            'frames_received': self._frames_received,
            'vad_chunks_processed': self._vad_chunks_processed,
            'total_audio_bytes': self._total_audio_bytes,
            'avg_transcription_ms': avg_transcription_ms,
            'threshold': self.threshold,
            'eager_silence_ms': self._eager_silence_ms,
            'final_silence_ms': self._final_silence_ms,
            'total_eager_eots': self._total_eager_eots,
            'total_eager_confirmed': self._total_eager_confirmed,
            'total_eager_cancelled': self._total_eager_cancelled,
            'device': self.device,
            'remote_url': self.cohere_transcribe_url,
            'http_session': 'requests' if self._http_session else 'urllib',
            'debug_log': DEBUG_LOG,
        }
