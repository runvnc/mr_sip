"""
Silero VAD + Cohere Transcribe STT Provider

Silero VAD runs locally (CPU-friendly). Cohere Transcribe runs on a remote GPU server.

Audio pipeline:
  ulaw 8kHz (SIP) -> Silero VAD (8kHz native) -> speech detection
  AGC normalization applied per chunk before VAD and speech buffering
  On speech start  -> fire barge-in callback (turn_resumed)
  Buffer ulaw during speech
  On speech end    -> POST ulaw bytes to COHERE_TRANSCRIBE_URL -> text -> emit final

Key tuning parameters (all configurable via stt_config dict or env vars):
  threshold              - VAD speech sensitivity (0.0-1.0, default 0.5)
  min_silence_duration_ms - silence needed to end utterance (default 600ms)
  speech_pad_ms          - padding added around speech (default 30ms)
  max_utterance_duration_s - hard cap on utterance length (default 30s)
  agc_target_rms         - AGC target RMS level (default 0.1, 0 to disable)
"""
import asyncio
import audioop
import logging
import os
import sys
import time
import traceback
from typing import Optional, Callable

import urllib.request
import json as _json
import numpy as np
import torch

from .base_stt import BaseSTTProvider, STTResult

logger = logging.getLogger(__name__)

# VAD chunk size at 8kHz: 256 samples = 256 bytes ulaw = 32ms
VAD_CHUNK_SAMPLES = 256
VAD_SAMPLE_RATE = 8000
COHERE_SAMPLE_RATE = 16000

# Dedicated debug log file - always written regardless of log level
DEBUG_LOG = '/tmp/silero_cohere_stt.log'


def _dlog(msg: str):
    """Write a timestamped line to the debug log file and also to logger.info."""
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    try:
        with open(DEBUG_LOG, 'a') as f:
            f.write(line + '\n')
            f.flush()
    except Exception:
        pass
    logger.info(msg)


class SileroCohereSTT(BaseSTTProvider):
    """
    Local VAD + remote ASR STT provider.

    Barge-in detection: Silero VAD fires on every speech onset.
    The turn_resumed_callback is called immediately, which halts AI audio
    output and cancels any in-progress LLM response.
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
        self.min_silence_duration_ms = int(
            os.getenv('SILERO_MIN_SILENCE_MS', str(min_silence_duration_ms))
        )
        self.speech_pad_ms = int(os.getenv('SILERO_SPEECH_PAD_MS', str(speech_pad_ms)))
        self.language = os.getenv('COHERE_TRANSCRIBE_LANGUAGE', language)
        self.cohere_model_id = os.getenv('COHERE_TRANSCRIBE_MODEL', cohere_model_id)
        self.max_utterance_duration_s = float(
            os.getenv('COHERE_MAX_UTTERANCE_S', str(max_utterance_duration_s))
        )

        # AGC: normalize each chunk to this RMS level (0 = disabled)
        self.agc_target_rms = float(
            os.getenv('SILERO_AGC_TARGET_RMS', '0.1')
        )

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.device = os.getenv('COHERE_TRANSCRIBE_DEVICE', self.device)

        self.cohere_transcribe_url = (
            cohere_transcribe_url
            or os.getenv('COHERE_TRANSCRIBE_URL', '')
        ).rstrip('/')

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
        self._frames_received: int = 0
        self._vad_chunks_processed: int = 0

        # Callbacks
        self._turn_resumed_callback: Optional[Callable] = None

        # Stats
        self._utterance_count: int = 0
        self._total_audio_bytes: int = 0
        self._transcription_times: list = []

        _dlog(
            f'SileroCohereSTT.__init__: threshold={self.threshold}, '
            f'min_silence={self.min_silence_duration_ms}ms, '
            f'speech_pad={self.speech_pad_ms}ms, '
            f'agc_target_rms={self.agc_target_rms}, '
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
            _dlog('SileroCohereSTT.start: no remote URL - loading Cohere model locally...')
            try:
                await asyncio.get_event_loop().run_in_executor(None, self._load_cohere_model)
                _dlog('SileroCohereSTT.start: local Cohere model loaded OK.')
            except Exception as e:
                _dlog(f'SileroCohereSTT.start: FATAL - failed to load local Cohere model: {e}')
                _dlog(traceback.format_exc())
                raise RuntimeError(
                    f'SileroCohereSTT: failed to load Cohere Transcribe model locally. '
                    f'Set COHERE_TRANSCRIBE_URL to use remote server instead. Error: {e}'
                ) from e

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
        self.vad_iterator = VADIterator(
            model=self.vad_model,
            threshold=self.threshold,
            sampling_rate=VAD_SAMPLE_RATE,
            min_silence_duration_ms=self.min_silence_duration_ms,
            speech_pad_ms=self.speech_pad_ms,
        )
        _dlog(f'_load_vad_model: VADIterator created (threshold={self.threshold}, '
              f'min_silence={self.min_silence_duration_ms}ms, sr={VAD_SAMPLE_RATE})')

    def _load_cohere_model(self):
        """Load Cohere Transcribe model locally (fallback when no remote URL)."""
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
        self.cohere_processor = AutoProcessor.from_pretrained(
            self.cohere_model_id, trust_remote_code=True)
        self.cohere_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.cohere_model_id,
            device_map=self.device,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            trust_remote_code=True,
        )
        self.cohere_model.eval()

    async def stop(self) -> None:
        """Stop the provider and release resources."""
        if not self.is_running:
            return
        self.is_running = False
        self._is_speaking = False
        self._vad_buffer = b''
        self._speech_buffer = b''
        if self.vad_iterator is not None:
            try:
                self.vad_iterator.reset_states()
            except Exception:
                pass
        _dlog(f'SileroCohereSTT stopped. Stats: frames={self._frames_received}, '
              f'vad_chunks={self._vad_chunks_processed}, utterances={self._utterance_count}')

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
                f'utterances={self._utterance_count}'
            )

        # Buffer audio during speech for transcription
        if self._is_speaking:
            self._speech_buffer += audio_bytes
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
        try:
            self._vad_chunks_processed += 1

            # Log first chunk to confirm VAD is receiving audio
            if self._vad_chunks_processed == 1:
                _dlog('_process_vad_chunk: first VAD chunk received - VAD is active')

            # ulaw -> PCM int16 -> float32 tensor
            pcm_bytes = audioop.ulaw2lin(chunk_bytes, 2)
            audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
            audio_float = audio_int16.astype(np.float32) / 32768.0

            # AGC: normalize to target RMS to handle low-volume phones
            if self.agc_target_rms > 0:
                rms = float(np.sqrt(np.mean(audio_float ** 2)))
                if rms > 1e-6:
                    gain = self.agc_target_rms / rms
                    # Cap gain at 20x to avoid amplifying pure noise
                    gain = min(gain, 20.0)
                    audio_float = np.clip(audio_float * gain, -1.0, 1.0)
                    # Re-encode normalized audio back to ulaw for speech buffer
                    norm_int16 = (audio_float * 32767).astype(np.int16)
                    chunk_bytes = audioop.lin2ulaw(norm_int16.tobytes(), 2)

            chunk_tensor = torch.from_numpy(audio_float)

            # VADIterator returns {'start': N} or {'end': N} or None
            result = self.vad_iterator(chunk_tensor, return_seconds=False)

            if result is None:
                return

            _dlog(f'_process_vad_chunk: VAD event: {result}')

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

        self._is_speaking = True
        self._speech_start_time = time.time()
        self._speech_buffer = b''
        _dlog(f'[VAD] Speech START (utterance #{self._utterance_count + 1})')

        if self._turn_resumed_callback is not None:
            try:
                _dlog('[VAD] Firing turn_resumed_callback (barge-in)')
                self._turn_resumed_callback()
            except Exception as e:
                _dlog(f'_on_speech_start: turn_resumed_callback error: {e}')

    async def _on_speech_end(self) -> None:
        """Called when Silero VAD detects end-of-speech. Transcribes buffered audio."""
        if not self._is_speaking:
            return

        self._is_speaking = False
        speech_duration = time.time() - self._speech_start_time
        speech_bytes = self._speech_buffer
        self._speech_buffer = b''

        _dlog(f'[VAD] Speech END: {speech_duration:.2f}s, {len(speech_bytes)} bytes buffered')

        if len(speech_bytes) < VAD_CHUNK_SAMPLES * 2:
            _dlog('[VAD] Speech segment too short (<512 bytes), skipping transcription')
            return

        t0 = time.time()
        _dlog(f'[TRANSCRIBE] Starting transcription of {len(speech_bytes)} bytes...')
        try:
            text = await asyncio.get_event_loop().run_in_executor(
                None, self._transcribe_ulaw, speech_bytes
            )
        except Exception as e:
            _dlog(f'[TRANSCRIBE] ERROR: {e}\n{traceback.format_exc()}')
            return

        elapsed = time.time() - t0
        self._transcription_times.append(elapsed)
        _dlog(f'[TRANSCRIBE] Done in {elapsed*1000:.0f}ms -> "{text}"')

        if not text or not text.strip():
            _dlog('[TRANSCRIBE] Empty result, skipping emit')
            return

        self._utterance_count += 1
        result = STTResult(
            text=text.strip(),
            is_final=True,
            is_eager_eot=False,
            confidence=0.95,
            timestamp=time.time(),
        )
        result.utterance_num = self._utterance_count
        _dlog(f'[EMIT] Final utterance #{self._utterance_count}: "{text.strip()}"')
        self._emit_final(result)

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    def _transcribe_ulaw(self, ulaw_bytes: bytes) -> str:
        """Route to remote or local transcription."""
        if self.cohere_transcribe_url:
            return self._transcribe_remote(ulaw_bytes)
        return self._transcribe_local(ulaw_bytes)

    def _transcribe_remote(self, ulaw_bytes: bytes) -> str:
        """POST ulaw bytes to the remote Cohere Transcribe HTTP server."""
        url = f'{self.cohere_transcribe_url}/transcribe?language={self.language}'
        _dlog(f'_transcribe_remote: POST {len(ulaw_bytes)} bytes to {url}')
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
        """Run Cohere Transcribe locally (fallback)."""
        if self.cohere_model is None:
            _dlog('_transcribe_local: ERROR - local model not loaded and no remote URL')
            return ''

        pcm_bytes = audioop.ulaw2lin(ulaw_bytes, 2)
        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0
        audio_16k = self._resample_2x(audio_float)

        inputs = self.cohere_processor(
            audio=audio_16k,
            sampling_rate=COHERE_SAMPLE_RATE,
            return_tensors='pt',
            language=self.language,
        )
        audio_chunk_index = inputs.pop('audio_chunk_index', None)
        inputs_on_device = {
            k: v.to(self.device) if hasattr(v, 'to') else v
            for k, v in inputs.items()
        }

        with torch.no_grad():
            outputs = self.cohere_model(**inputs_on_device)

        try:
            text = self.cohere_processor.decode(
                outputs,
                skip_special_tokens=True,
                audio_chunk_index=audio_chunk_index,
                language=self.language,
            )
            if isinstance(text, list):
                text = text[0] if text else ''
        except Exception:
            try:
                ids = outputs.logits.argmax(dim=-1) if hasattr(outputs, 'logits') else outputs
                text = self.cohere_processor.batch_decode(ids, skip_special_tokens=True)[0]
            except Exception as e2:
                _dlog(f'_transcribe_local: decode fallback failed: {e2}')
                text = ''

        return text

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
            'min_silence_duration_ms': self.min_silence_duration_ms,
            'device': self.device,
            'remote_url': self.cohere_transcribe_url,
            'debug_log': DEBUG_LOG,
        }
