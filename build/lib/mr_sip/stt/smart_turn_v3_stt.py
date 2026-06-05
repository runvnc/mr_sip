"""
Smart Turn v3 + Silero VAD + Cohere Transcribe STT Provider

Silero VAD for speech start (barge-in). Smart Turn v3 ONNX model polled every 80ms
for turn-completion detection. Cohere Transcribe for ASR.

Architecture:
  ulaw 8kHz (SIP) -> Silero VAD (speech start only) -> barge-in callback
  During speech: buffer ulaw, resample to 16kHz, poll Smart Turn v3 every 80ms
  Smart Turn v3 outputs turn_complete probability (0-1)
  When prob > threshold -> transcribe -> emit final
  Fallback: if silence exceeds max_silence_poll_ms, transcribe anyway

Key env vars:
  SMART_TURN_MODEL_PATH - path to ONNX model (auto-download if missing)
  SMART_TURN_POLL_MS - polling interval (default 80)
  SMART_TURN_THRESHOLD - probability threshold (default 0.5)
  SMART_TURN_MAX_SILENCE_POLL_MS - fallback silence timeout (default 2000)
  SMART_TURN_MIN_SPEECH_MS - minimum speech before accepting turn detection (default 500)
  SMART_TURN_DEVICE - 'cuda' or 'cpu' (default 'cuda')
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

VAD_CHUNK_SAMPLES = 256
VAD_SAMPLE_RATE = 8000
COHERE_SAMPLE_RATE = 16000
SMART_TURN_SAMPLE_RATE = 16000
SMART_TURN_MAX_DURATION_S = 8.0

DEBUG_LOG = '/tmp/smart_turn_v3_stt.log'
E2E_LATENCY_LOG = '/tmp/sip_e2e_latency.log'


def _dlog(msg: str):
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


def _e2e_log(event: str, utterance_num: int = 0, **kwargs):
    now = datetime.now()
    ts = now.strftime('%Y-%m-%d %H:%M:%S') + f'.{now.microsecond // 1000:03d}'
    pc = time.perf_counter()
    extra = ' '.join(f'{k}={v}' for k, v in kwargs.items())
    line = f'[{ts}] [E2E] {event} perf_counter={pc:.6f} utterance={utterance_num} {extra}'
    try:
        with open(E2E_LATENCY_LOG, 'a') as f:
            f.write(line + '\n')
            f.flush()
    except Exception:
        pass
    logger.info(f'[E2E] {event} utterance={utterance_num} {extra}')


class SmartTurnV3STT(BaseSTTProvider):
    """
    Silero VAD for speech start + Smart Turn v3 polling for turn-end + Cohere Transcribe.

    Silero VAD detects speech onset (barge-in). Smart Turn v3 ONNX model is polled
    every poll_ms during speech to detect turn completion. When Smart Turn signals
    turn complete, audio is sent to Cohere Transcribe and emitted as final.
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
        print('Smart turn init!!!')
        # Silero VAD params (speech start only)
        self.threshold = float(os.getenv('SILERO_VAD_THRESHOLD', str(threshold)))
        self.speech_pad_ms = int(os.getenv('SILERO_SPEECH_PAD_MS', str(speech_pad_ms)))

        # Smart Turn v3 params
        self._poll_ms = int(os.getenv('SMART_TURN_POLL_MS', '80'))
        self._turn_threshold = float(os.getenv('SMART_TURN_THRESHOLD', '0.5'))
        self._max_silence_poll_ms = int(os.getenv('SMART_TURN_MAX_SILENCE_POLL_MS', '2000'))
        self._min_speech_ms = int(os.getenv('SMART_TURN_MIN_SPEECH_MS', '500'))
        self._min_end_silence_ms = int(os.getenv('SMART_TURN_MIN_END_SILENCE_MS', '125'))
        self._model_path = os.getenv('SMART_TURN_MODEL_PATH', '')
        self._smart_turn_device = os.getenv('SMART_TURN_DEVICE', 'cuda')

        # Cohere Transcribe params
        self.language = os.getenv('COHERE_TRANSCRIBE_LANGUAGE', language)
        self.cohere_model_id = os.getenv('COHERE_TRANSCRIBE_MODEL', cohere_model_id)
        self.max_utterance_duration_s = float(
            os.getenv('COHERE_MAX_UTTERANCE_S', str(max_utterance_duration_s))
        )
        self.cohere_transcribe_url = (
            cohere_transcribe_url
            or os.getenv('COHERE_TRANSCRIBE_URL', '')
        ).rstrip('/')

        # Full-buffer normalization before transcription
        self.transcribe_target_rms = float(
            os.getenv('SILERO_TRANSCRIBE_TARGET_RMS', '0.15')
        )

        # Pre-roll buffer
        preroll_ms = int(os.getenv('SILERO_PREROLL_MS', '300'))
        self._preroll_chunks = max(0, preroll_ms // 32)
        self._preroll_buffer: deque = deque(maxlen=self._preroll_chunks) if self._preroll_chunks > 0 else deque(maxlen=0)

        # Device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # HTTP session for Cohere
        self._http_session = None
        if _requests is not None and self.cohere_transcribe_url:
            self._http_session = _requests.Session()
            self._http_session.headers.update({
                'Content-Type': 'application/octet-stream',
                'User-Agent': 'mr-sip/1.0',
            })

        # Models
        self.vad_model = None
        self._ort_session = None
        self._feature_extractor = None

        # VAD state (Silero for speech start only)
        self._vad_speech_active = False
        self._vad_silence_chunks = 0
        self._vad_silence_chunks_needed = 0

        # Audio state
        self._vad_buffer: bytes = b''
        self._speech_buffer: bytes = b''
        self._is_speaking: bool = False
        self._speech_start_time: float = 0.0
        self._last_speech_audio_time: float = 0.0
        self._frames_received: int = 0
        self._vad_chunks_processed: int = 0

        # Smart Turn polling state
        self._poll_task: Optional[asyncio.Task] = None
        self._poll_active: bool = False
        self._turn_detected: bool = False
        self._silence_start_time: Optional[float] = None
        self._last_poll_time: float = 0.0

        # Callbacks
        self._turn_resumed_callback: Optional[Callable] = None

        # Stats
        self._utterance_count: int = 0
        self._total_audio_bytes: int = 0
        self._last_turn_was_eager: bool = False
        self._transcription_times: list = []
        self._smart_turn_inference_times: list = []
        self._total_turns_detected: int = 0
        self._total_fallback_transcriptions: int = 0

        _dlog(
            f'SmartTurnV3STT.__init__: threshold={self.threshold}, '
            f'poll_ms={self._poll_ms}, turn_threshold={self._turn_threshold}, '
            f'max_silence_poll_ms={self._max_silence_poll_ms}, '
            f'min_speech_ms={self._min_speech_ms}, '
            f'min_end_silence_ms={self._min_end_silence_ms}, '
            f'model_path="{self._model_path or "(auto-download)"}", '
            f'smart_turn_device={self._smart_turn_device}, '
            f'remote_url="{self.cohere_transcribe_url or "(none)"}", '
            f'language={self.language}'
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Load Silero VAD, Smart Turn ONNX model, validate Cohere URL."""
        if self.is_running:
            _dlog('SmartTurnV3STT.start: already running, skipping')
            return

        _dlog('SmartTurnV3STT.start: loading Silero VAD model...')
        try:
            await asyncio.get_event_loop().run_in_executor(None, self._load_vad_model)
        except Exception as e:
            _dlog(f'SmartTurnV3STT.start: FATAL - failed to load Silero VAD: {e}')
            raise RuntimeError(f'Failed to load Silero VAD: {e}') from e
        _dlog('SmartTurnV3STT.start: Silero VAD loaded OK.')

        _dlog('SmartTurnV3STT.start: loading Smart Turn v3 ONNX model...')
        try:
            await asyncio.get_event_loop().run_in_executor(None, self._load_smart_turn_model)
        except Exception as e:
            _dlog(f'SmartTurnV3STT.start: FATAL - failed to load Smart Turn v3: {e}')
            raise RuntimeError(f'Failed to load Smart Turn v3: {e}') from e
        _dlog('SmartTurnV3STT.start: Smart Turn v3 loaded OK.')

        if self.cohere_transcribe_url:
            _dlog(f'SmartTurnV3STT.start: checking remote Cohere Transcribe at {self.cohere_transcribe_url} ...')
            try:
                health_url = f'{self.cohere_transcribe_url}/health'
                req = urllib.request.Request(
                    health_url,
                    headers={'User-Agent': 'mr-sip/1.0'},
                    method='GET',
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    body = _json.loads(resp.read())
                _dlog(f'SmartTurnV3STT.start: remote health OK: {body}')
            except Exception as e:
                raise RuntimeError(
                    f'Cohere Transcribe server not reachable at {self.cohere_transcribe_url}/health. Error: {e}'
                ) from e
        else:
            raise RuntimeError('COHERE_TRANSCRIBE_URL is not set.')

        self.is_running = True
        _dlog('SmartTurnV3STT.start: provider started and ready.')

    def _load_vad_model(self):
        """Load Silero VAD (blocking, run in executor)."""
        try:
            print("loading silero vad")
            from silero_vad import load_silero_vad
        except ImportError:
            raise ImportError('silero-vad package not installed. Run: pip install silero-vad')
        self.vad_model = load_silero_vad(onnx=False)
        self._vad_silence_chunks_needed = int(os.getenv('SILERO_MIN_SILENCE_MS', '600')) // 32
        _dlog(f'_load_vad_model: threshold={self.threshold}, silence_chunks_needed={self._vad_silence_chunks_needed}')

    def _load_smart_turn_model(self):
        """Load Smart Turn v3 ONNX model. Download if needed."""
        print("loading smart turn v3")
        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor

        model_path = self._model_path
        if not model_path or not os.path.exists(model_path):
            model_path = self._download_smart_turn_model()
            self._model_path = model_path

        _dlog(f'_load_smart_turn_model: loading ONNX from {model_path}')

        providers = ['CPUExecutionProvider']
        if self._smart_turn_device == 'cuda':
            try:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            except Exception:
                _dlog('_load_smart_turn_model: CUDA not available, falling back to CPU')

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._ort_session = ort.InferenceSession(model_path, sess_options=so, providers=providers)
        self._feature_extractor = WhisperFeatureExtractor(chunk_length=SMART_TURN_MAX_DURATION_S)
        _dlog(f'_load_smart_turn_model: loaded, providers={self._ort_session.get_providers()}')

    def _download_smart_turn_model(self) -> str:
        """Download Smart Turn v3 ONNX model from HuggingFace."""
        from huggingface_hub import hf_hub_download

        model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        os.makedirs(model_dir, exist_ok=True)

        _dlog(f'_download_smart_turn_model: downloading from pipecat-ai/smart-turn-v3 to {model_dir}...')
        model_path = hf_hub_download(
            repo_id='pipecat-ai/smart-turn-v3',
            filename='smart-turn-v3.1-gpu.onnx',
            local_dir=model_dir,
            local_dir_use_symlinks=False,
        )
        _dlog(f'_download_smart_turn_model: downloaded to {model_path}')
        return model_path

    async def stop(self) -> None:
        """Stop the provider and release resources."""
        if not self.is_running:
            return
        self.is_running = False
        self._is_speaking = False
        self._poll_active = False
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            self._poll_task = None
        self._vad_buffer = b''
        self._speech_buffer = b''
        self._preroll_buffer.clear()
        if self._http_session is not None:
            try:
                self._http_session.close()
            except Exception:
                pass
        _dlog(f'SmartTurnV3STT stopped. Stats: frames={self._frames_received}, '
              f'utterances={self._utterance_count}, turns_detected={self._total_turns_detected}, '
              f'fallbacks={self._total_fallback_transcriptions}')

    # ------------------------------------------------------------------
    # Audio ingestion
    # ------------------------------------------------------------------

    async def add_audio_bytes(self, audio_bytes: bytes) -> None:
        """Feed raw ulaw 8kHz bytes from SIP into the pipeline."""
        if not self.is_running:
            return

        self._frames_received += 1
        self._total_audio_bytes += len(audio_bytes)

        if self._frames_received % 500 == 0:
            _dlog(
                f'add_audio_bytes: heartbeat - frames={self._frames_received}, '
                f'is_speaking={self._is_speaking}, speech_buf={len(self._speech_buffer)}B, '
                f'poll_active={self._poll_active}, utterances={self._utterance_count}'
            )

        # Buffer audio during speech for transcription
        if self._is_speaking:
            self._speech_buffer += audio_bytes
            self._last_speech_audio_time = time.perf_counter()
            max_bytes = int(self.max_utterance_duration_s * VAD_SAMPLE_RATE)
            if len(self._speech_buffer) > max_bytes:
                _dlog(f'add_audio_bytes: utterance exceeded {self.max_utterance_duration_s}s limit, forcing end')
                await self._on_turn_complete()
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
    # VAD processing (Silero - speech start only)
    # ------------------------------------------------------------------

    async def _process_vad_chunk(self, chunk_bytes: bytes) -> None:
        """Run one 256-byte (32ms) ulaw chunk through Silero VAD.

        Only used for speech start detection (barge-in). Turn-end is handled
        by Smart Turn v3 polling.
        """
        try:
            self._vad_chunks_processed += 1

            if self._vad_chunks_processed == 1:
                _dlog('_process_vad_chunk: first VAD chunk received')

            # ulaw -> PCM int16 -> float32 tensor
            pcm_bytes = audioop.ulaw2lin(chunk_bytes, 2)
            audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
            audio_float = audio_int16.astype(np.float32) / 32768.0
            chunk_tensor = torch.from_numpy(audio_float)

            prob = self.vad_model(chunk_tensor, VAD_SAMPLE_RATE).item()

            # Speech start detection only
            if not self._vad_speech_active:
                if prob >= self.threshold:
                    self._vad_speech_active = True
                    self._vad_silence_chunks = 0
                    await self._on_speech_start()
            else:
                # Track silence for fallback, but don't trigger speech end
                if prob < self.threshold:
                    self._vad_silence_chunks += 1
                else:
                    self._vad_silence_chunks = 0

            # Always update pre-roll buffer
            if not self._is_speaking:
                self._preroll_buffer.append(chunk_bytes)

        except Exception as e:
            _dlog(f'_process_vad_chunk: ERROR: {e}\n{traceback.format_exc()}')

    async def _on_speech_start(self) -> None:
        """Called when Silero VAD detects speech onset. Fire barge-in, start polling."""
        if self._is_speaking:
            return

        self._is_speaking = True
        self._speech_start_time = time.perf_counter()
        self._last_speech_audio_time = time.perf_counter()
        self._speech_buffer = b''
        self._turn_detected = False
        self._silence_start_time = None

        # Prepend pre-roll buffer
        if self._preroll_buffer:
            preroll_bytes = b''.join(self._preroll_buffer)
            self._speech_buffer = preroll_bytes
            _dlog(f'[VAD] Pre-roll: prepended {len(preroll_bytes)} bytes')

        _dlog(f'[VAD] Speech START (utterance #{self._utterance_count + 1})')

        # Fire barge-in callback
        if self._turn_resumed_callback is not None:
            try:
                _dlog('[VAD] Firing turn_resumed_callback (barge-in)')
                self._turn_resumed_callback()
            except Exception as e:
                _dlog(f'_on_speech_start: turn_resumed_callback error: {e}')

        # Start Smart Turn polling
        self._start_polling()

    # ------------------------------------------------------------------
    # Smart Turn v3 polling
    # ------------------------------------------------------------------

    def _start_polling(self):
        """Start the Smart Turn polling loop."""
        if self._poll_active:
            return
        self._poll_active = True
        self._poll_task = asyncio.ensure_future(self._poll_loop())
        _dlog(f'[SMART_TURN] Polling started (every {self._poll_ms}ms)')

    def _stop_polling(self):
        """Stop the Smart Turn polling loop.

        Only sets _poll_active=False so the loop exits naturally.
        Do NOT cancel _poll_task here - _on_turn_complete is called
        from inside the poll loop, so cancelling would kill the
        in-progress transcription.
        """
        self._poll_active = False
        _dlog('[SMART_TURN] Polling stopped')

    async def _poll_loop(self):
        """Poll Smart Turn v3 every poll_ms during speech."""
        try:
            while self._poll_active and self.is_running:
                await asyncio.sleep(self._poll_ms / 1000.0)

                if not self._is_speaking or self._turn_detected:
                    continue

                # Minimum speech buffer check (basic sanity)
                min_bytes_basic = VAD_CHUNK_SAMPLES * 4  # 1024 bytes
                if len(self._speech_buffer) < min_bytes_basic:
                    continue

                # Minimum speech duration before accepting turn detection
                speech_elapsed_ms = (time.perf_counter() - self._speech_start_time) * 1000
                if speech_elapsed_ms < self._min_speech_ms:
                    if self._frames_received % 50 == 0:
                        _dlog(f'[SMART_TURN] Waiting for min speech: {speech_elapsed_ms:.0f}ms < {self._min_speech_ms}ms')
                    continue

                self._last_poll_time = time.perf_counter()

                # Check if we've been silent too long (fallback)
                silence_duration = (time.perf_counter() - self._last_speech_audio_time) * 1000
                if silence_duration > self._max_silence_poll_ms:
                    _dlog(f'[SMART_TURN] Fallback: silence for {silence_duration:.0f}ms > {self._max_silence_poll_ms}ms, forcing turn complete')
                    self._total_fallback_transcriptions += 1
                    self._last_turn_was_eager = False
                    await self._on_turn_complete()
                    continue

                # Run Smart Turn inference
                try:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, self._run_smart_turn_inference
                    )
                except Exception as e:
                    _dlog(f'[SMART_TURN] Inference error: {e}')
                    continue

                prob = result['probability']
                prediction = result['prediction']

                if self._utterance_count < 3 or self._frames_received % 50 == 0:
                    _dlog(f'[SMART_TURN] Poll: prob={prob:.3f}, prediction={prediction}, '
                          f'speech_buf={len(self._speech_buffer)}B, '
                          f'silence={silence_duration:.0f}ms, '
                          f'vad_silence_chunks={self._vad_silence_chunks}')

                if prediction == 1 and prob >= self._turn_threshold:
                    # Require minimum silence before accepting turn completion.
                    # This prevents mid-sentence splits where the speaker just
                    # pauses briefly between words.
                    silence_at_end_ms = self._vad_silence_chunks * 32
                    if silence_at_end_ms >= self._min_end_silence_ms:
                        _dlog(f'[SMART_TURN] Turn DETECTED: prob={prob:.3f}, '
                              f'end_silence={silence_at_end_ms}ms >= {self._min_end_silence_ms}ms')
                        self._last_turn_was_eager = True
                        self._turn_detected = True
                        self._total_turns_detected += 1
                        await self._on_turn_complete()
                    else:
                        _dlog(f'[SMART_TURN] Turn predicted (prob={prob:.3f}) but '
                              f'insufficient silence: {silence_at_end_ms}ms < {self._min_end_silence_ms}ms, waiting...')

        except asyncio.CancelledError:
            _dlog('[SMART_TURN] Poll loop cancelled')
        except Exception as e:
            _dlog(f'[SMART_TURN] Poll loop error: {e}\n{traceback.format_exc()}')
        finally:
            self._poll_task = None

    def _run_smart_turn_inference(self) -> dict:
        """Run Smart Turn v3 ONNX inference on current speech buffer.

        Returns dict with 'prediction' (0 or 1) and 'probability' (float).
        """
        t0 = time.perf_counter()

        # Convert ulaw speech buffer to 16kHz float32
        pcm_bytes = audioop.ulaw2lin(self._speech_buffer, 2)
        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0

        # Resample 8kHz -> 16kHz via linear interpolation
        audio_16k = self._resample_2x(audio_float)

        # Truncate to last 8 seconds or pad to 8 seconds
        max_samples = int(SMART_TURN_MAX_DURATION_S * SMART_TURN_SAMPLE_RATE)
        if len(audio_16k) > max_samples:
            audio_16k = audio_16k[-max_samples:]
        elif len(audio_16k) < max_samples:
            padding = max_samples - len(audio_16k)
            audio_16k = np.pad(audio_16k, (padding, 0), mode='constant', constant_values=0)

        # Extract features using WhisperFeatureExtractor
        inputs = self._feature_extractor(
            audio_16k,
            sampling_rate=SMART_TURN_SAMPLE_RATE,
            return_tensors='np',
            padding='max_length',
            max_length=max_samples,
            truncation=True,
            do_normalize=True,
        )
        input_features = inputs.input_features.squeeze(0).astype(np.float32)
        input_features = np.expand_dims(input_features, axis=0)

        # Run ONNX inference
        outputs = self._ort_session.run(None, {'input_features': input_features})
        probability = outputs[0][0].item()
        prediction = 1 if probability > 0.5 else 0

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._smart_turn_inference_times.append(elapsed_ms)

        return {'prediction': prediction, 'probability': probability}

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
    # Turn completion -> transcribe
    # ------------------------------------------------------------------

    async def _on_turn_complete(self) -> None:
        """Smart Turn says turn complete (or fallback). Transcribe and emit final."""
        if not self._is_speaking:
            return

        self._is_speaking = False
        self._vad_speech_active = False
        self._stop_polling()

        turn_end_time = time.perf_counter()
        speech_duration = turn_end_time - self._speech_start_time
        speech_bytes = self._speech_buffer
        self._speech_buffer = b''

        _dlog(f'[TURN_COMPLETE] Speech duration: {speech_duration:.2f}s, {len(speech_bytes)} bytes')
        _e2e_log('TURN_COMPLETE', utterance_num=self._utterance_count + 1,
                 speech_duration_s=f'{speech_duration:.2f}', bytes=len(speech_bytes))

        if len(speech_bytes) < VAD_CHUNK_SAMPLES * 2:
            _dlog('[TURN_COMPLETE] Speech segment too short, skipping transcription')
            return

        # Full-buffer normalization
        speech_bytes = self._normalize_buffer(speech_bytes)

        t0 = time.perf_counter()
        _dlog(f'[TRANSCRIBE] Starting transcription of {len(speech_bytes)} bytes...')
        try:
            text = await asyncio.get_event_loop().run_in_executor(
                None, self._transcribe_ulaw, speech_bytes
            )
        except Exception as e:
            _dlog(f'[TRANSCRIBE] ERROR: {e}\n{traceback.format_exc()}')
            return

        elapsed_pc = time.perf_counter() - t0
        self._transcription_times.append(elapsed_pc)
        _dlog(f'[TRANSCRIBE] Done in {elapsed_pc*1000:.0f}ms -> "{text}"')
        _e2e_log('TRANSCRIBE_DONE', utterance_num=self._utterance_count + 1,
                 transcribe_ms=f'{elapsed_pc*1000:.0f}')

        if not text or not text.strip():
            _dlog('[TRANSCRIBE] Empty result, skipping emit')
            return

        text = text.strip()
        self._utterance_count += 1

        result = STTResult(
            text=text,
            is_final=True,
            is_eager_eot=self._last_turn_was_eager,
            confidence=0.9,
            timestamp=time.time(),
        )
        result.utterance_num = self._utterance_count
        _dlog(f'[EMIT] Final #{self._utterance_count}: "{text}"')
        self._emit_final(result)

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    def _transcribe_ulaw(self, ulaw_bytes: bytes) -> str:
        """POST ulaw bytes to remote Cohere Transcribe server."""
        url = f'{self.cohere_transcribe_url}/transcribe?language={self.language}'
        _dlog(f'_transcribe_ulaw: POST {len(ulaw_bytes)} bytes to {url}')

        if self._http_session is not None:
            try:
                resp = self._http_session.post(url, data=ulaw_bytes, timeout=30)
                body = resp.json()
                text = body.get('text', '')
                _dlog(f'_transcribe_ulaw: response: {body}')
                return text
            except Exception as e:
                _dlog(f'_transcribe_ulaw (requests): FAILED: {e}')
                return ''

        req = urllib.request.Request(
            url,
            data=ulaw_bytes,
            headers={'Content-Type': 'application/octet-stream', 'User-Agent': 'mr-sip/1.0'},
            method='POST',
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = _json.loads(resp.read())
                text = body.get('text', '')
                _dlog(f'_transcribe_ulaw: response: {body}')
                return text
        except Exception as e:
            _dlog(f'_transcribe_ulaw: FAILED: {e}')
            return ''

    def _normalize_buffer(self, ulaw_bytes: bytes) -> bytes:
        """Normalize full speech buffer to target RMS level."""
        if self.transcribe_target_rms <= 0:
            return ulaw_bytes
        pcm_bytes = audioop.ulaw2lin(ulaw_bytes, 2)
        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(audio_float ** 2)))
        if rms < 1e-6:
            return ulaw_bytes
        gain = min(self.transcribe_target_rms / rms, 10.0)
        audio_float = np.clip(audio_float * gain, -1.0, 1.0)
        norm_int16 = (audio_float * 32767).astype(np.int16)
        return audioop.lin2ulaw(norm_int16.tobytes(), 2)

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
        avg_st_inference_ms = (
            sum(self._smart_turn_inference_times) / len(self._smart_turn_inference_times)
            if self._smart_turn_inference_times else 0
        )
        return {
            'provider': 'smart_turn_v3',
            'vad_model': 'silero_vad_v6',
            'turn_model': 'smart_turn_v3',
            'asr_model': self.cohere_model_id,
            'is_running': self.is_running,
            'utterance_count': self._utterance_count,
            'frames_received': self._frames_received,
            'vad_chunks_processed': self._vad_chunks_processed,
            'total_audio_bytes': self._total_audio_bytes,
            'avg_transcription_ms': avg_transcription_ms,
            'avg_smart_turn_inference_ms': avg_st_inference_ms,
            'poll_ms': self._poll_ms,
            'turn_threshold': self._turn_threshold,
            'max_silence_poll_ms': self._max_silence_poll_ms,
            'total_turns_detected': self._total_turns_detected,
            'total_fallback_transcriptions': self._total_fallback_transcriptions,
            'device': self.device,
            'smart_turn_device': self._smart_turn_device,
            'remote_url': self.cohere_transcribe_url,
            'debug_log': DEBUG_LOG,
        }
