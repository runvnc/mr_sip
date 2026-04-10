"""
Silero VAD + Cohere Transcribe STT Provider

Local, zero-cloud-dependency speech-to-text for SIP calls.

Audio pipeline:
  ulaw 8kHz (SIP) -> Silero VAD (8kHz native) -> speech detection
  On speech start  -> fire barge-in callback (turn_resumed)
  Buffer ulaw during speech
  On speech end    -> ulaw -> PCM float32 -> resample 8->16kHz
                  -> Cohere Transcribe -> text -> emit final

Key tuning parameters (all configurable via stt_config dict or env vars):
  threshold              - VAD speech sensitivity (0.0-1.0, default 0.5)
  min_silence_duration_ms - silence needed to end utterance (default 400ms)
  speech_pad_ms          - padding added around speech (default 30ms)
  max_utterance_duration_s - hard cap on utterance length (default 30s)
"""
import asyncio
import audioop
import logging
import os
import time
from typing import Optional, Callable

import numpy as np
import torch

from .base_stt import BaseSTTProvider, STTResult

logger = logging.getLogger(__name__)

# VAD chunk size at 8kHz: 256 samples = 256 bytes ulaw = 32ms
VAD_CHUNK_SAMPLES = 256
VAD_SAMPLE_RATE = 8000
COHERE_SAMPLE_RATE = 16000


class SileroCohereSTT(BaseSTTProvider):
    """
    Local VAD + ASR STT provider.

    Barge-in detection: Silero VAD fires on every speech onset.
    The turn_resumed_callback is called immediately, which halts AI audio
    output and cancels any in-progress LLM response - same behaviour as
    Deepgram Flux TurnResumed.
    """

    def __init__(
        self,
        sample_rate: int = 8000,
        threshold: float = 0.5,
        min_silence_duration_ms: int = 400,
        speech_pad_ms: int = 30,
        language: str = 'en',
        cohere_model_id: str = 'CohereLabs/cohere-transcribe-03-2026',
        device: Optional[str] = None,
        max_utterance_duration_s: float = 30.0,
        **kwargs,
    ):
        """
        Args:
            sample_rate: Input audio sample rate (must be 8000 for SIP ulaw).
            threshold: Silero VAD speech probability threshold (0-1, default 0.5).
                       Lower = more sensitive, higher = fewer false triggers.
            min_silence_duration_ms: Silence duration (ms) required to end an
                       utterance. Controls end-of-speech latency (default 400ms).
            speech_pad_ms: Padding added to start/end of detected speech (default 30ms).
            language: Language code for Cohere Transcribe (default 'en').
            cohere_model_id: HuggingFace model ID for Cohere Transcribe.
            device: Torch device ('cuda', 'cpu', or None for auto-detect).
            max_utterance_duration_s: Hard cap on utterance buffer length (default 30s).
        """
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

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.device = os.getenv('COHERE_TRANSCRIBE_DEVICE', self.device)

        # Models - loaded lazily in start()
        self.vad_model = None
        self.vad_iterator = None
        self.cohere_model = None
        self.cohere_processor = None

        # Audio state
        self._vad_buffer: bytes = b''       # accumulate bytes until VAD chunk size
        self._speech_buffer: bytes = b''    # ulaw bytes buffered during speech
        self._is_speaking: bool = False
        self._speech_start_time: float = 0.0

        # Callbacks
        self._turn_resumed_callback: Optional[Callable] = None

        # Stats
        self._utterance_count: int = 0
        self._total_audio_bytes: int = 0
        self._transcription_times: list = []

        logger.info(
            f'SileroCohereSTT init: threshold={self.threshold}, '
            f'min_silence={self.min_silence_duration_ms}ms, '
            f'device={self.device}, model={self.cohere_model_id}'
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Load Silero VAD and Cohere Transcribe models."""
        if self.is_running:
            logger.warning('SileroCohereSTT already running')
            return

        logger.info('Loading Silero VAD model...')
        await asyncio.get_event_loop().run_in_executor(None, self._load_vad_model)
        logger.info('Silero VAD loaded.')

        logger.info(f'Loading Cohere Transcribe model ({self.cohere_model_id})...')
        await asyncio.get_event_loop().run_in_executor(None, self._load_cohere_model)
        logger.info('Cohere Transcribe loaded.')

        self.is_running = True
        logger.info('SileroCohereSTT started.')

    def _load_vad_model(self):
        """Load Silero VAD (blocking, run in executor)."""
        try:
            from silero_vad import load_silero_vad, VADIterator
        except ImportError:
            raise ImportError(
                'silero-vad package not installed. '
                'Run: pip install silero-vad'
            )

        self.vad_model = load_silero_vad(onnx=False)
        self.vad_iterator = VADIterator(
            model=self.vad_model,
            threshold=self.threshold,
            sampling_rate=VAD_SAMPLE_RATE,
            min_silence_duration_ms=self.min_silence_duration_ms,
            speech_pad_ms=self.speech_pad_ms,
        )

    def _load_cohere_model(self):
        """Load Cohere Transcribe model (blocking, run in executor)."""
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

        self.cohere_processor = AutoProcessor.from_pretrained(
            self.cohere_model_id,
            trust_remote_code=True,
        )
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
        logger.info('SileroCohereSTT stopped.')

    # ------------------------------------------------------------------
    # Audio ingestion
    # ------------------------------------------------------------------

    async def add_audio_bytes(self, audio_bytes: bytes) -> None:
        """
        Feed raw ulaw 8kHz bytes from SIP into the VAD pipeline.

        Called for every RTP frame (~20ms / 160 bytes).
        Accumulates bytes into 256-byte (32ms) VAD chunks.
        """
        if not self.is_running:
            return

        self._total_audio_bytes += len(audio_bytes)

        # Buffer audio during speech for transcription
        if self._is_speaking:
            self._speech_buffer += audio_bytes
            # Hard cap: prevent runaway buffering
            max_bytes = int(self.max_utterance_duration_s * VAD_SAMPLE_RATE)
            if len(self._speech_buffer) > max_bytes:
                logger.warning(
                    f'Utterance exceeded {self.max_utterance_duration_s}s limit, '
                    'forcing end-of-speech.'
                )
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
        """
        Compatibility: accept float32 numpy audio and convert to ulaw bytes.
        Not the primary path for SIP (use add_audio_bytes instead).
        """
        if not self.is_running:
            return
        # Convert float32 -> int16 -> ulaw
        audio_int16 = (np.clip(audio_chunk, -1.0, 1.0) * 32767).astype(np.int16)
        pcm_bytes = audio_int16.tobytes()
        ulaw_bytes = audioop.lin2ulaw(pcm_bytes, 2)
        await self.add_audio_bytes(ulaw_bytes)

    # ------------------------------------------------------------------
    # VAD processing
    # ------------------------------------------------------------------

    async def _process_vad_chunk(self, chunk_bytes: bytes) -> None:
        """
        Run one 256-byte (32ms) ulaw chunk through Silero VAD.
        Fires speech start/end events.
        """
        try:
            # ulaw -> PCM int16 -> float32 tensor
            pcm_bytes = audioop.ulaw2lin(chunk_bytes, 2)
            audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
            audio_float = audio_int16.astype(np.float32) / 32768.0
            chunk_tensor = torch.from_numpy(audio_float)

            # VADIterator returns {'start': N} or {'end': N} or None
            result = self.vad_iterator(chunk_tensor, return_seconds=False)

            if result is None:
                return

            if 'start' in result:
                await self._on_speech_start()
            elif 'end' in result:
                await self._on_speech_end()

        except Exception as e:
            logger.error(f'Error in VAD chunk processing: {e}')

    async def _on_speech_start(self) -> None:
        """
        Called when Silero VAD detects speech onset.

        Fires the barge-in (turn_resumed) callback unconditionally.
        This halts AI audio output and cancels any in-progress LLM response.
        The caller (sip_client_v2._handle_turn_resumed) is safe to call even
        when the AI is not speaking - it will be a no-op in that case.
        """
        if self._is_speaking:
            return  # already in a speech segment

        logger.info('[VAD] Speech start detected')
        self._is_speaking = True
        self._speech_start_time = time.time()
        self._speech_buffer = b''  # reset buffer for new utterance

        # Fire barge-in callback - stops AI audio and cancels draft response
        if self._turn_resumed_callback is not None:
            try:
                self._turn_resumed_callback()
            except Exception as e:
                logger.error(f'Error in turn_resumed_callback: {e}')

    async def _on_speech_end(self) -> None:
        """
        Called when Silero VAD detects end-of-speech.
        Transcribes the buffered audio and emits a final STTResult.
        """
        if not self._is_speaking:
            return

        logger.info('[VAD] Speech end detected')
        self._is_speaking = False

        speech_bytes = self._speech_buffer
        self._speech_buffer = b''

        if len(speech_bytes) < VAD_CHUNK_SAMPLES * 2:
            logger.debug('Speech segment too short, skipping transcription.')
            return

        # Transcribe in executor to avoid blocking the event loop
        t0 = time.time()
        try:
            text = await asyncio.get_event_loop().run_in_executor(
                None, self._transcribe_ulaw, speech_bytes
            )
        except Exception as e:
            logger.error(f'Transcription error: {e}')
            return

        elapsed = time.time() - t0
        self._transcription_times.append(elapsed)
        logger.info(f'[TRANSCRIBE] {elapsed*1000:.0f}ms -> "{text}"')

        if not text or not text.strip():
            logger.debug('Empty transcription, skipping.')
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
        self._emit_final(result)

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    def _transcribe_ulaw(self, ulaw_bytes: bytes) -> str:
        """
        Convert ulaw 8kHz bytes -> float32 PCM -> resample to 16kHz
        -> Cohere Transcribe -> text.

        Runs synchronously (called via run_in_executor).
        """
        # ulaw -> PCM int16
        pcm_bytes = audioop.ulaw2lin(ulaw_bytes, 2)
        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0

        # Resample 8kHz -> 16kHz (simple 2x linear interpolation)
        audio_16k = self._resample_2x(audio_float)

        # Run Cohere Transcribe
        inputs = self.cohere_processor(
            audio=audio_16k,
            sampling_rate=COHERE_SAMPLE_RATE,
            return_tensors='pt',
            language=self.language,
        )

        # Extract audio_chunk_index before moving tensors to device
        audio_chunk_index = inputs.pop('audio_chunk_index', None)

        # Move inputs to device
        inputs_on_device = {
            k: v.to(self.device) if hasattr(v, 'to') else v
            for k, v in inputs.items()
        }

        with torch.no_grad():
            outputs = self.cohere_model(**inputs_on_device)

        # Decode
        try:
            text = self.cohere_processor.decode(
                outputs,
                skip_special_tokens=True,
                audio_chunk_index=audio_chunk_index,
                language=self.language,
            )
            # decode() may return a list or a string depending on version
            if isinstance(text, list):
                text = text[0] if text else ''
        except Exception:
            # Fallback: try batch_decode on the raw output tensor
            try:
                if hasattr(outputs, 'logits'):
                    ids = outputs.logits.argmax(dim=-1)
                else:
                    ids = outputs
                text = self.cohere_processor.batch_decode(
                    ids, skip_special_tokens=True
                )[0]
            except Exception as e2:
                logger.error(f'Decode fallback failed: {e2}')
                text = ''

        return text

    @staticmethod
    def _resample_2x(audio: np.ndarray) -> np.ndarray:
        """
        Upsample audio by 2x using linear interpolation (8kHz -> 16kHz).
        Fast and dependency-free.
        """
        n = len(audio)
        # Interleave original samples with interpolated midpoints
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
        """
        Set the barge-in callback.
        Called by sip_client_v2 during STT setup.
        Fired on every speech onset to halt AI audio and cancel draft responses.
        """
        self._turn_resumed_callback = callback

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
            'total_audio_bytes': self._total_audio_bytes,
            'avg_transcription_ms': avg_transcription_ms,
            'threshold': self.threshold,
            'min_silence_duration_ms': self.min_silence_duration_ms,
            'device': self.device,
        }
