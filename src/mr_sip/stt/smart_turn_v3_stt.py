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
  SMART_TURN_MIN_SPEECH_MS - minimum speech before accepting turn detection (default 250)
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
from .whisper_features import compute_whisper_log_mel_features
from .bargein_gate import BargeInGate

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

DEADAIR_LOG = '/tmp/sip_deadair.log'


def _deadair_log(event: str, utterance_num: int = 0, **kwargs):
    """Append an STT-side dead-air trigger marker to DEADAIR_LOG."""
    now = datetime.now()
    ts = now.strftime('%Y-%m-%d %H:%M:%S') + f'.{now.microsecond // 1000:03d}'
    pc = time.perf_counter()
    extra = ' '.join(f'{k}={v}' for k, v in kwargs.items())
    line = f'[{ts}] [DEADAIR] {event} perf_counter={pc:.6f} utterance={utterance_num} {extra}'
    try:
        with open(DEADAIR_LOG, 'a') as f:
            f.write(line + '\n')
            f.flush()
    except Exception:
        pass
    logger.info(f'[DEADAIR-STT] {event} utterance={utterance_num} {extra}')


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
        threshold: float = 0.3,
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
        self._min_speech_ms = int(os.getenv('SMART_TURN_MIN_SPEECH_MS', '250'))
        self._semantic_check_silence_ms = int(os.getenv(
            'SMART_TURN_SEMANTIC_CHECK_SILENCE_MS', os.getenv('SMART_TURN_MIN_END_SILENCE_MS', '384')
        ))
        self._final_confirm_silence_ms = int(os.getenv('SMART_TURN_FINAL_CONFIRM_SILENCE_MS', '640'))
        self._eager_enabled = os.getenv('SMART_TURN_EAGER_ENABLED', '1').lower() not in ('0', 'false', 'no')
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
        preroll_ms = int(os.getenv('SILERO_PREROLL_MS', '500'))
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

        # VAD state (Silero for speech start only)
        self._vad_speech_active = False
        self._vad_silence_chunks = 0
        self._vad_silence_chunks_needed = 0

        # ------------------------------------------------------------------
        # Barge-in / foreground-vs-background gate (shared BargeInGate).
        # Silero is primary; level is a second opinion. Path A fires on the
        # first near-end voiced frame (zero added latency). Path B rescues a
        # SUSTAINED loud segment Silero misses (e.g. a clipped greeting). BG
        # (voiced but clearly quieter than the near-end reference) and NS never
        # onset, so quiet background cross-talk neither halts the AI nor gets
        # buffered/transcribed/injected. The gate is fed (silero_prob, raw_rms)
        # on EVERY 32ms chunk so its near-end reference + noise floor stay
        # coherent across the whole call.
        # ------------------------------------------------------------------
        self._gate = BargeInGate(
            vad_threshold=self.threshold,
            frame_ms=32,
            rel_level_db=float(os.getenv('BARGE_IN_REL_LEVEL_DB', '15')),
            min_rms=float(os.getenv('BARGE_IN_MIN_RMS', '0.01')),
            ref_attack_alpha=float(os.getenv('BARGE_IN_REF_ATTACK_ALPHA', '0.4')),
            ref_decay_alpha=float(os.getenv('BARGE_IN_REF_DECAY_ALPHA', '0.05')),
            onset_voiced_frames=int(os.getenv('SILERO_ONSET_VOICED_FRAMES', '1')),
            level_window_ms=int(os.getenv('BARGE_IN_LEVEL_WINDOW_MS', '160')),
            rescue_enabled=os.getenv('BARGE_IN_RESCUE_ENABLED', '1').lower() not in ('0', 'false', 'no'),
            rescue_snr_db=float(os.getenv('BARGE_IN_RESCUE_SNR_DB', '12')),
            rescue_sustain_ms=int(os.getenv('BARGE_IN_RESCUE_SUSTAIN_MS', '160')),
            noise_alpha=float(os.getenv('BARGE_IN_NOISE_ALPHA', '0.05')),
        )
        # How quiet BACKGROUND (BG) speech is treated for turn-END timing while
        # a turn is active (BARGE_IN_BG_DURING_TURN):
        #   'silence' (default) - BG counts toward end-silence just like true
        #       non-speech, so overlapping background that is below the near-end
        #       reference lets the turn endpoint at the caller's words even with
        #       NO real pause (caller "Hello?" + quieter overlapping woman ->
        #       endpoints on "Hello?"). The discriminator is purely level: tune
        #       BARGE_IN_REL_LEVEL_DB (lower = stricter = more of a somewhat-
        #       quieter overlap is treated as background).
        #   'neutral'  - BG neither resets nor accrues silence (frozen). Best
        #       protects a quiet FG tail, but will NOT endpoint over continuous
        #       overlapping background (no true pause).
        #   'voiced'   - legacy: any voiced frame (incl. background) keeps the
        #       turn alive.
        self._bg_during_turn = os.getenv('BARGE_IN_BG_DURING_TURN', 'silence').strip().lower()
        if self._bg_during_turn not in ('silence', 'neutral', 'voiced'):
            self._bg_during_turn = 'silence'
        # Trim trailing background past the last near-end frame (see below) is
        # active whenever we are not in legacy 'voiced' mode.
        self._bg_ends_turn = self._bg_during_turn != 'voiced'
        # Turn-END level strictness, DECOUPLED from the onset gate. The onset
        # gate (BARGE_IN_REL_LEVEL_DB, 15dB) must stay permissive so a genuinely
        # quiet near-end caller can still barge in. But for ENDING a turn over
        # overlapping cross-talk we want a stricter split: a frame counts as
        # near-end (keeps the turn alive) only if it is within this many dB of
        # the near-end reference. Anything quieter accrues end-silence. Default
        # 9dB (stricter than onset) so moderate, only-slightly-quieter
        # background ends the turn out of the box - no env override needed.
        # This is safe against cutting a quiet FG tail because endpointing ALSO
        # requires a sustained (>= SMART_TURN_SEMANTIC_CHECK_SILENCE_MS) below-
        # near-end run AND Smart Turn predicting semantic completion. <=0 falls
        # back to the gate's own FG/BG label.
        self._turn_end_rel_db = float(os.getenv('BARGE_IN_TURN_END_REL_LEVEL_DB', '9'))
        # On turn end, trim trailing audio beyond the last near-end (FG) frame
        # (plus this pad) so any background that bled into the buffer before the
        # endpoint fired is not transcribed. <0 disables trimming.
        self._bg_trim_pad_ms = int(os.getenv('BARGE_IN_BG_TRIM_PAD_MS', '200'))
        # ulaw byte offset (within _speech_buffer) of the last near-end frame.
        self._last_fg_speech_len = 0

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

        # Eager EOT state. Smart Turn can emit a cancellable eager EOT after
        # semantic completion, then confirm final after longer silence.
        self._eager_pending: bool = False
        self._eager_text: str = ''
        self._eager_utterance_num: int = 0
        self._eager_audio_len: int = 0

        # Stats
        self._utterance_count: int = 0
        self._total_audio_bytes: int = 0
        self._transcription_times: list = []
        self._smart_turn_inference_times: list = []
        self._total_turns_detected: int = 0
        self._total_fallback_transcriptions: int = 0
        self._total_eager_eots: int = 0
        self._total_eager_confirmed: int = 0
        self._total_eager_cancelled: int = 0

        _dlog(
            f'SmartTurnV3STT.__init__: threshold={self.threshold}, '
            f'poll_ms={self._poll_ms}, turn_threshold={self._turn_threshold}, '
            f'max_silence_poll_ms={self._max_silence_poll_ms}, '
            f'min_speech_ms={self._min_speech_ms}, '
            f'semantic_check_silence_ms={self._semantic_check_silence_ms}, '
            f'final_confirm_silence_ms={self._final_confirm_silence_ms}, '
            f'eager_enabled={self._eager_enabled}, '
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

        # ORT CUDA EP can silently fail to locate CUDA/cuDNN libs from inside
        # MindRoot's venv, then fall back to CPU.  Preload first when available
        # (ORT >= 1.21) before the ONNX session is created.
        try:
            if hasattr(ort, 'preload_dlls'):
                ort.preload_dlls()
                _dlog('_load_smart_turn_model: ort.preload_dlls() OK')
        except Exception as e:
            _dlog(f'_load_smart_turn_model: ort.preload_dlls() failed/nonfatal: {e}')

        model_path = self._model_path
        if not model_path or not os.path.exists(model_path):
            model_path = self._download_smart_turn_model()
            self._model_path = model_path

        _dlog(f'_load_smart_turn_model: loading ONNX from {model_path}')

        providers = ['CPUExecutionProvider']
        if self._smart_turn_device == 'cuda':
            providers = [
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                    'cudnn_conv_algo_search': 'DEFAULT',
                    'do_copy_in_default_stream': '1',
                    'use_tf32': '1',
                }),
                'CPUExecutionProvider',
            ]

        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._ort_session = ort.InferenceSession(model_path, sess_options=so, providers=providers)
        _dlog(f'_load_smart_turn_model: loaded, providers={self._ort_session.get_providers()}')

        if self._smart_turn_device == 'cuda' and 'CUDAExecutionProvider' not in self._ort_session.get_providers():
            raise RuntimeError(
                f'SMART_TURN_DEVICE=cuda requested but ONNX Runtime active providers are {self._ort_session.get_providers()}; refusing silent CPU fallback'
            )

    def _smart_turn_runtime_label(self) -> str:
        """Return a short runtime label for the active Smart Turn ONNX provider.

        This is logged at the start of each user turn so runtime GPU/CPU status is
        visible near the actual endpointing events, not just during startup.
        """
        if self._ort_session is None:
            return f'requested={self._smart_turn_device}, active=not_loaded, providers=[]'

        try:
            providers = list(self._ort_session.get_providers())
        except Exception as e:
            return f'requested={self._smart_turn_device}, active=unknown, providers_error={e!r}'

        if 'CUDAExecutionProvider' in providers:
            active = 'cuda'
        elif 'CPUExecutionProvider' in providers:
            active = 'cpu'
        else:
            active = 'unknown'
        return f'requested={self._smart_turn_device}, active={active}, providers={providers}'

    def _download_smart_turn_model(self) -> str:
        """Download Smart Turn v3 ONNX model from HuggingFace."""
        from huggingface_hub import hf_hub_download

        model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        os.makedirs(model_dir, exist_ok=True)

        filename = os.getenv('SMART_TURN_MODEL_FILENAME', 'smart-turn-v3.2-gpu.onnx')
        _dlog(f'_download_smart_turn_model: downloading {filename} from pipecat-ai/smart-turn-v3 to {model_dir}...')
        model_path = hf_hub_download(
            repo_id='pipecat-ai/smart-turn-v3',
            filename=filename,
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
        self._eager_pending = False
        self._eager_text = ''
        self._eager_utterance_num = 0
        self._eager_audio_len = 0
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

            # Raw per-chunk RMS on the SAME audio the VAD sees (no AGC). Used by
            # the barge-in level gate and to build the near-end reference level.
            rms = float(np.sqrt(np.mean(audio_float ** 2)))
            voiced = prob >= self.threshold

            # Feed the shared gate on EVERY chunk so its near-end reference and
            # noise floor stay coherent across the whole call.
            gate = self._gate.process(prob, rms)
            label = gate['label']

            # Speech start detection only.
            if not self._vad_speech_active:
                if gate['barge_in']:
                    # Foreground onset (Path A near-end, or Path B loud rescue).
                    now_pc = time.perf_counter()
                    self._vad_speech_active = True
                    self._vad_silence_chunks = 0
                    self._last_speech_audio_time = now_pc
                    rel_str = f"{gate['rel_db']:.1f}" if gate['rel_db'] is not None else 'n/a'
                    _dlog(f"[BARGE-IN] Onset accepted ({gate['reason']}): "
                          f"rms={rms:.5f}, near_end_ref={gate['near_end_ref']}, "
                          f"rel_db={rel_str}, snr_db={gate['snr_db']:.1f}, prob={prob:.3f}")
                    _e2e_log('VAD_ONSET_ACCEPTED',
                             utterance_num=self._utterance_count + 1,
                             reason=gate['reason'], rel_db=rel_str,
                             snr_db=f"{gate['snr_db']:.1f}", prob=f'{prob:.3f}')
                    await self._on_speech_start()
                    # Include the VAD-triggering chunk itself. add_audio_bytes()
                    # only appends new RTP audio while _is_speaking was already
                    # true at function entry, so without this the first voiced
                    # 32ms chunk is dropped from the transcription buffer.
                    # Matters most for very short responses like "Blue".
                    if self._is_speaking:
                        self._speech_buffer += chunk_bytes
                elif label == self._gate.BG:
                    # Quiet background cross-talk: ignore outright. No halt, no
                    # buffer, no transcription. Log occasionally to avoid spam.
                    if self._vad_chunks_processed % 50 == 0:
                        rel_str = f"{gate['rel_db']:.1f}" if gate['rel_db'] is not None else 'n/a'
                        _dlog(f'[BARGE-IN] Ignoring background frame: '
                              f"rms={rms:.5f}, near_end_ref={gate['near_end_ref']}, "
                              f'rel_db={rel_str}, prob={prob:.3f}')
            else:
                # During an active turn we keep buffering ALL audio to the Smart
                # Turn endpoint (so a quiet intra-utterance FG tail is never
                # truncated), but turn-END timing uses the gate label:
                #   FG  -> near-end speech: reset silence, keep the turn alive,
                #          and remember the buffer position (for trailing trim).
                #   NS  -> true non-speech: accrue end-silence.
                #   BG  -> quieter-than-near-end background cross-talk. Handled
                #          per BARGE_IN_BG_DURING_TURN: 'silence' (default) makes
                #          it accrue end-silence so the turn endpoints at the
                #          caller's words even when background OVERLAPS with no
                #          real pause; 'neutral' freezes the timer; 'voiced' is
                #          legacy keep-alive. The FG/BG split is purely by level
                #          vs the near-end reference, so BARGE_IN_REL_LEVEL_DB is
                #          the knob: LOWER it to treat a smaller level gap as
                #          background (ends the turn sooner over louder/closer
                #          cross-talk); RAISE it to be more permissive.
                # Do NOT use raw RTP arrival time for _last_speech_audio_time:
                # RTP packets keep arriving during silence, which made the
                # fallback silence timer useless. Update it on near-end only.
                # Stricter turn-end level test (decoupled from onset). A frame
                # keeps the turn alive only if it is near-end AND within
                # _turn_end_rel_db of the near-end reference; anything quieter
                # accrues end-silence so overlapping cross-talk ends the turn.
                rel_db = gate['rel_db']
                near_end_level = (
                    self._turn_end_rel_db <= 0
                    or rel_db is None
                    or rel_db >= -self._turn_end_rel_db
                )
                if self._bg_during_turn == 'voiced':
                    # Legacy: any voiced frame (incl. background) keeps the turn.
                    is_near_end = voiced
                    is_silence = not voiced
                elif self._bg_during_turn == 'neutral':
                    # Only near-end keeps the turn; only true non-speech accrues
                    # silence; quieter background is frozen.
                    is_near_end = (label == self._gate.FG and near_end_level)
                    is_silence = (label == self._gate.NS)
                else:  # 'silence' (default)
                    # Everything not clearly near-end (NS, BG, or an FG frame
                    # quieter than _turn_end_rel_db) accrues end-silence, so the
                    # turn endpoints at the caller's words even when background
                    # overlaps with no real pause.
                    is_near_end = (label == self._gate.FG and near_end_level)
                    is_silence = not is_near_end

                if is_near_end:
                    self._vad_silence_chunks = 0
                    if self._is_speaking:
                        self._last_speech_audio_time = time.perf_counter()
                        self._last_fg_speech_len = len(self._speech_buffer)
                elif is_silence:
                    self._vad_silence_chunks += 1
                # else: BG-neutral -> leave _vad_silence_chunks frozen.

                # If an eager EOT has been emitted and speech resumes before
                # final confirmation, cancel the pending eager state. Only a
                # FOREGROUND resume cancels: quiet background (BG/NS) must not
                # cancel a legitimate eager EOT.
                if voiced and self._eager_pending:
                    if label == self._gate.FG:
                        self._total_eager_cancelled += 1
                        _dlog(f'[EAGER] Cancelled Smart Turn eager EOT #{self._eager_utterance_num} - user resumed speaking')
                        _deadair_log('STT_EAGER_CANCEL_FG_RESUME',
                                     utterance_num=self._eager_utterance_num,
                                     rms=f'{rms:.5f}')
                        self._eager_pending = False
                        self._eager_text = ''
                        self._eager_utterance_num = 0
                        self._eager_audio_len = 0
                        if self._turn_resumed_callback is not None:
                            self._turn_resumed_callback()
                    else:
                        rel_str = f"{gate['rel_db']:.1f}" if gate['rel_db'] is not None else 'n/a'
                        _dlog(f'[EAGER] Ignoring {label} resume during eager #{self._eager_utterance_num} '
                              f'(rms={rms:.5f}, rel_db={rel_str})')

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

        # Log speech start for e2e profiling
        _e2e_log('VAD_SPEECH_START', utterance_num=self._utterance_count + 1,
                 threshold=self.threshold)

        # Prepend pre-roll buffer
        if self._preroll_buffer:
            preroll_bytes = b''.join(self._preroll_buffer)
            self._speech_buffer = preroll_bytes
            _dlog(f'[VAD] Pre-roll: prepended {len(preroll_bytes)} bytes')

        # Baseline near-end buffer position for trailing-background trim. The
        # onset itself is near-end; subsequent FG chunks push this forward.
        self._last_fg_speech_len = len(self._speech_buffer)

        _dlog(f'[VAD] Speech START (utterance #{self._utterance_count + 1}) - '
              f'Smart Turn runtime: {self._smart_turn_runtime_label()}')

        # Fire barge-in callback
        if self._turn_resumed_callback is not None:
            try:
                _dlog('[VAD] Firing turn_resumed_callback (barge-in)')
                _deadair_log('STT_SPEECH_START_BARGE_IN',
                             utterance_num=self._utterance_count + 1)
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
        """Poll Smart Turn v3 after VAD has observed silence.

        This follows the intended Smart Turn usage more closely than continuous
        polling during active speech: wait for a short VAD silence, run semantic
        endpointing on the full current turn, optionally emit a cancellable eager
        EOT, then confirm final after a longer silence window.
        """
        try:
            while self._poll_active and self.is_running:
                await asyncio.sleep(self._poll_ms / 1000.0)

                if not self._is_speaking or self._turn_detected:
                    continue

                min_bytes_basic = VAD_CHUNK_SAMPLES * 4  # 1024 bytes
                if len(self._speech_buffer) < min_bytes_basic:
                    continue

                speech_elapsed_ms = (time.perf_counter() - self._speech_start_time) * 1000
                if speech_elapsed_ms < self._min_speech_ms:
                    if self._frames_received % 50 == 0:
                        _dlog(f'[SMART_TURN] Waiting for min speech: {speech_elapsed_ms:.0f}ms < {self._min_speech_ms}ms')
                    continue

                self._last_poll_time = time.perf_counter()

                silence_duration = (time.perf_counter() - self._last_speech_audio_time) * 1000
                silence_at_end_ms = self._vad_silence_chunks * 32

                if silence_duration > self._max_silence_poll_ms:
                    _dlog(f'[SMART_TURN] Fallback: silence for {silence_duration:.0f}ms > {self._max_silence_poll_ms}ms, forcing turn complete')
                    self._total_fallback_transcriptions += 1
                    await self._on_turn_complete(reason='max_silence_fallback')
                    continue

                # If eager was already emitted, final-confirm once the longer
                # silence window has elapsed. No second ASR is needed; final uses
                # the eager text and sip_client_v2 can skip duplicate agent send.
                if self._eager_pending:
                    if silence_at_end_ms >= self._final_confirm_silence_ms:
                        _dlog(f'[SMART_TURN] Final confirming eager #{self._eager_utterance_num}: silence={silence_at_end_ms}ms >= {self._final_confirm_silence_ms}ms')
                        self._turn_detected = True
                        self._total_turns_detected += 1
                        await self._on_turn_complete(reason='eager_final_confirmed')
                    continue

                # Do not run semantic endpointing until VAD has actually seen a
                # pause. This avoids the old behavior where Smart Turn was asked
                # mid-word/mid-speech and often returned high completion scores.
                if silence_at_end_ms < self._semantic_check_silence_ms:
                    if self._utterance_count < 3 or self._frames_received % 50 == 0:
                        _dlog(f'[SMART_TURN] Waiting for semantic silence: {silence_at_end_ms}ms < {self._semantic_check_silence_ms}ms')
                    continue

                try:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, self._run_smart_turn_inference
                    )
                except Exception as e:
                    _dlog(f'[SMART_TURN] Inference error: {e}')
                    continue

                prob = result['probability']
                prediction = result['prediction']

                _dlog(f'[SMART_TURN] Poll: prob={prob:.3f}, prediction={prediction}, '
                      f'speech_buf={len(self._speech_buffer)}B, '
                      f'silence={silence_duration:.0f}ms, '
                      f'end_silence={silence_at_end_ms}ms, '
                      f'semantic_ms={self._semantic_check_silence_ms}, '
                      f'final_ms={self._final_confirm_silence_ms}')

                if prediction == 1 and prob >= self._turn_threshold:
                    if silence_at_end_ms >= self._final_confirm_silence_ms or not self._eager_enabled:
                        _dlog(f'[SMART_TURN] Turn DETECTED final: prob={prob:.3f}, end_silence={silence_at_end_ms}ms')
                        self._turn_detected = True
                        self._total_turns_detected += 1
                        await self._on_turn_complete(reason='semantic_final')
                    else:
                        _dlog(f'[SMART_TURN] Eager candidate: prob={prob:.3f}, '
                              f'end_silence={silence_at_end_ms}ms < final {self._final_confirm_silence_ms}ms')
                        await self._emit_eager_eot(prob=prob, silence_at_end_ms=silence_at_end_ms)
                else:
                    _dlog(f'[SMART_TURN] Semantic incomplete: prob={prob:.3f}, prediction={prediction}, waiting...')

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

        # Extract features using the Pipecat-style numpy Whisper log-mel path.
        input_features = compute_whisper_log_mel_features(audio_16k, do_normalize=True)
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


    async def _emit_eager_eot(self, prob: float, silence_at_end_ms: int) -> None:
        """Transcribe current buffer and emit a cancellable eager EOT partial."""
        if self._eager_pending or not self._is_speaking:
            return

        speech_bytes = bytes(self._speech_buffer)
        # Drop any trailing background cross-talk that leaked in before this
        # eager endpoint so it is not transcribed / merged into the eager text.
        speech_bytes = self._trim_trailing_background(speech_bytes)
        if len(speech_bytes) < VAD_CHUNK_SAMPLES * 2:
            _dlog('[EAGER] Speech segment too short, skipping eager EOT')
            return

        speech_bytes = self._normalize_buffer(speech_bytes)
        eager_pc = time.perf_counter()
        self._last_vad_eager_end_pc = eager_pc
        self._last_user_speech_end_pc = eager_pc - (silence_at_end_ms / 1000.0)
        _e2e_log('SMART_TURN_EAGER_END', utterance_num=self._utterance_count + 1,
                 prob=f'{prob:.3f}', silence_ms=silence_at_end_ms)

        t0 = time.perf_counter()
        _dlog(f'[EAGER] Starting transcription of {len(speech_bytes)} bytes (prob={prob:.3f}, silence={silence_at_end_ms}ms)...')
        try:
            text = await asyncio.get_event_loop().run_in_executor(
                None, self._transcribe_ulaw, speech_bytes
            )
        except Exception as e:
            _dlog(f'[EAGER] TRANSCRIBE ERROR: {e}\n{traceback.format_exc()}')
            return

        elapsed_pc = time.perf_counter() - t0
        self._transcription_times.append(elapsed_pc)
        _dlog(f'[EAGER] Transcribe done in {elapsed_pc*1000:.0f}ms -> "{text}"')
        _e2e_log('TRANSCRIBE_DONE', utterance_num=self._utterance_count + 1,
                 transcribe_ms=f'{elapsed_pc*1000:.0f}', eager=True)

        if not text or not text.strip():
            _dlog('[EAGER] Empty transcription, skipping eager emit')
            return

        # ASR takes time. If the user resumed while transcription was in flight,
        # do not emit a stale eager EOT after barge-in has already happened.
        current_silence_at_end_ms = self._vad_silence_chunks * 32
        if current_silence_at_end_ms < silence_at_end_ms:
            self._total_eager_cancelled += 1
            _dlog(f'[EAGER] Skipping stale eager EOT: speech resumed during ASR '
                  f'(current_silence={current_silence_at_end_ms}ms < eager_silence={silence_at_end_ms}ms)')
            return

        text = text.strip()
        self._utterance_count += 1
        self._total_eager_eots += 1

        result = STTResult(
            text=text,
            is_final=False,
            is_eager_eot=True,
            confidence=0.8,
            timestamp=time.time(),
        )
        result.utterance_num = self._utterance_count
        _dlog(f'[EMIT] Eager EOT #{self._utterance_count}: "{text}"')
        self._emit_partial(result)

        self._eager_pending = True
        self._eager_text = text
        self._eager_utterance_num = self._utterance_count
        self._eager_audio_len = len(speech_bytes)

    # ------------------------------------------------------------------
    # Turn completion -> transcribe
    # ------------------------------------------------------------------

    async def _on_turn_complete(self, reason: str = 'semantic_final') -> None:
        """Turn complete. Emit final, reusing eager text if available."""
        if not self._is_speaking and not self._eager_pending:
            return

        was_eager_pending = self._eager_pending
        self._is_speaking = False
        self._vad_speech_active = False
        self._stop_polling()

        self._last_vad_eager_end_pc = time.perf_counter()
        self._last_user_speech_end_pc = self._last_vad_eager_end_pc

        turn_end_time = time.perf_counter()
        speech_duration = turn_end_time - self._speech_start_time
        speech_bytes = self._speech_buffer
        self._speech_buffer = b''

        _dlog(f'[TURN_COMPLETE] reason={reason} speech_duration={speech_duration:.2f}s, {len(speech_bytes)} bytes')
        _e2e_log('TURN_COMPLETE', utterance_num=(self._eager_utterance_num or self._utterance_count + 1),
                 speech_duration_s=f'{speech_duration:.2f}', bytes=len(speech_bytes), reason=reason)

        if was_eager_pending:
            text = self._eager_text
            utterance_num = self._eager_utterance_num
            self._eager_pending = False
            self._eager_text = ''
            self._eager_utterance_num = 0
            self._eager_audio_len = 0
            self._total_eager_confirmed += 1

            if not text:
                _dlog('[TURN_COMPLETE] Eager pending but eager text empty, skipping final')
                return

            result = STTResult(
                text=text,
                is_final=True,
                is_eager_eot=False,
                confidence=0.95,
                timestamp=time.time(),
            )
            result.utterance_num = utterance_num
            _dlog(f'[EMIT] Final (confirmed eager) #{utterance_num}: "{text}"')
            self._emit_final(result)
            _e2e_log('VAD_FINAL_CONFIRMED', utterance_num=utterance_num, reason=reason)
            return

        # Drop any trailing background cross-talk that leaked in before endpoint.
        speech_bytes = self._trim_trailing_background(speech_bytes)

        if len(speech_bytes) < VAD_CHUNK_SAMPLES * 2:
            _dlog('[TURN_COMPLETE] Speech segment too short, skipping transcription')
            return

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
            is_eager_eot=False,
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
        return self._normalize_buffer_impl(ulaw_bytes)

    def _trim_trailing_background(self, speech_bytes: bytes) -> bytes:
        """Trim audio past the last near-end (FG) frame (+ pad) so background
        cross-talk that bled into the buffer before the turn endpointed is not
        transcribed. Controlled by BARGE_IN_BG_ENDS_TURN / BARGE_IN_BG_TRIM_PAD_MS.
        """
        if not self._bg_ends_turn or self._bg_trim_pad_ms < 0:
            return speech_bytes
        if self._last_fg_speech_len <= 0:
            return speech_bytes
        pad = int(self._bg_trim_pad_ms / 1000.0 * VAD_SAMPLE_RATE)  # ulaw 1B/sample
        cut = min(len(speech_bytes), self._last_fg_speech_len + pad)
        if cut < len(speech_bytes):
            trimmed = len(speech_bytes) - cut
            _dlog(f'[TRIM] Trailing background trimmed: {trimmed} bytes '
                  f'({trimmed / VAD_SAMPLE_RATE * 1000:.0f}ms) after last FG '
                  f'(last_fg_len={self._last_fg_speech_len}, pad_ms={self._bg_trim_pad_ms})')
            return speech_bytes[:cut]
        return speech_bytes

    def _normalize_buffer_impl(self, ulaw_bytes: bytes) -> bytes:
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
            'semantic_check_silence_ms': self._semantic_check_silence_ms,
            'final_confirm_silence_ms': self._final_confirm_silence_ms,
            'eager_enabled': self._eager_enabled,
            'total_turns_detected': self._total_turns_detected,
            'total_fallback_transcriptions': self._total_fallback_transcriptions,
            'total_eager_eots': self._total_eager_eots,
            'total_eager_confirmed': self._total_eager_confirmed,
            'total_eager_cancelled': self._total_eager_cancelled,
            'device': self.device,
            'smart_turn_device': self._smart_turn_device,
            'remote_url': self.cohere_transcribe_url,
            'debug_log': DEBUG_LOG,
        }
