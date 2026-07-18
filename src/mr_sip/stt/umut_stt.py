"""Kyutai streaming ASR and semantic-pause turn detector for SIP.

This adapts the STT/turn-taking design from Kyutai's MIT-licensed Unmute
project while retaining MindRoot as the agent and LLM runtime. The official
moshi-server performs Mimi tokenization, streaming ASR, and pause prediction.
"""

import asyncio
import audioop
import logging
import math
import os
import time
from datetime import datetime
from typing import Callable, Optional

import msgpack
import numpy as np
import websockets

from .base_stt import BaseSTTProvider, STTResult

logger = logging.getLogger(__name__)

MODEL_SAMPLE_RATE = 24000
FRAME_SAMPLES = 1920
FRAME_TIME_SEC = FRAME_SAMPLES / MODEL_SAMPLE_RATE
DEFAULT_PATH = "/api/asr-streaming"
DEBUG_LOG = "/tmp/umut_stt.log"
UNINTERRUPTIBLE_BY_VAD_TIME_SEC = 3.0


def _as_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value, default):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _dlog(event: str, **fields) -> None:
    now = datetime.now()
    stamp = now.strftime("%Y-%m-%d %H:%M:%S") + f".{now.microsecond // 1000:03d}"
    suffix = " ".join(f"{key}={value}" for key, value in fields.items())
    try:
        with open(DEBUG_LOG, "a") as handle:
            handle.write(f"[{stamp}] {event}{(' ' + suffix) if suffix else ''}\n")
            handle.flush()
    except Exception:
        pass


class _EMA:
    """Unmute-compatible asymmetric exponential moving average."""

    def __init__(self, attack_time=0.01, release_time=0.01, initial_value=1.0):
        self.attack_time = attack_time
        self.release_time = release_time
        self.value = initial_value

    def update(self, dt: float, new_value: float) -> float:
        half_life = self.attack_time if new_value > self.value else self.release_time
        alpha = 1.0 - math.exp(-dt / half_life * math.log(2.0))
        self.value = float((1.0 - alpha) * self.value + alpha * new_value)
        return self.value


class UmutSTT(BaseSTTProvider):
    """Continuous Kyutai STT using its integrated learned pause detector."""

    def __init__(
        self,
        sample_rate=8000,
        stt_url=None,
        api_key=None,
        end_threshold=0.6,
        speech_threshold=0.4,
        delay_sec=0.5,
        queue_frames=250,
        vad_interruption=True,
        **kwargs,
    ):
        super().__init__(sample_rate=_as_int(sample_rate, 8000))
        base_url = stt_url or os.getenv("UMUT_STT_URL", "ws://127.0.0.1:8090")
        if not base_url.endswith(DEFAULT_PATH):
            base_url = base_url.rstrip("/") + DEFAULT_PATH
        self.stt_url = base_url
        self.api_key = api_key or os.getenv("UMUT_API_KEY", "public_token")
        self.end_threshold = _as_float(end_threshold, 0.6)
        self.speech_threshold = _as_float(speech_threshold, 0.4)
        self.delay_sec = _as_float(delay_sec, 0.5)
        self.queue_frames = _as_int(queue_frames, 250)
        self.vad_interruption = _as_bool(vad_interruption, True)

        self.websocket = None
        self._audio_queue = None
        self._sender_task = None
        self._receiver_task = None
        self._turn_resumed_callback: Optional[Callable] = None
        self._bot_speaking = False

        self._model_audio = np.empty(0, dtype=np.float32)
        self._rate_state = None
        self._pause = _EMA(initial_value=1.0)
        self._steps_to_ignore = 12
        self._current_time = -self.delay_sec
        self._turn_words = []
        self._turn_started = False
        self._speech_signal_active = False
        self._flushing = False
        self._flush_until = None
        self._utterance_count = 0
        self._last_user_speech_end_pc = 0.0
        self._last_strong_near_end_pc = 0.0

        self._frames_received = 0
        self._frames_sent = 0
        self._frames_dropped = 0
        self._words_received = 0
        self._pause_events = 0
        self._interruptions_word = 0
        self._interruptions_vad = 0
        self._started_pc = None

    async def start(self) -> None:
        if self.is_running:
            return
        self._audio_queue = asyncio.Queue(maxsize=max(10, self.queue_frames))
        _dlog("UMUT_CONNECT_BEGIN", url=self.stt_url, sample_rate=self.sample_rate)
        self.websocket = await websockets.connect(
            self.stt_url,
            additional_headers={"kyutai-api-key": self.api_key},
            max_size=None,
            ping_interval=20,
            ping_timeout=20,
        )
        first = msgpack.unpackb(await asyncio.wait_for(self.websocket.recv(), 30))
        if first.get("type") == "Error":
            await self.websocket.close()
            self.websocket = None
            raise RuntimeError(f"Umut STT server error: {first.get('message')}")
        if first.get("type") != "Ready":
            await self.websocket.close()
            self.websocket = None
            raise RuntimeError(f"Umut expected Ready, got {first!r}")

        self.is_running = True
        self._started_pc = time.perf_counter()
        self._sender_task = asyncio.create_task(self._sender_loop(), name="umut-stt-send")
        self._receiver_task = asyncio.create_task(
            self._receiver_loop(), name="umut-stt-receive"
        )
        _dlog(
            "UMUT_READY",
            end_threshold=self.end_threshold,
            speech_threshold=self.speech_threshold,
            delay_sec=self.delay_sec,
            vad_interruption=self.vad_interruption,
        )

    async def stop(self) -> None:
        if not self.is_running and self.websocket is None:
            return
        self.is_running = False
        tasks = [task for task in (self._sender_task, self._receiver_task) if task]
        for task in tasks:
            task.cancel()
        if self.websocket is not None:
            try:
                await self.websocket.close()
            except Exception:
                pass
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.websocket = None
        self._sender_task = None
        self._receiver_task = None
        _dlog("UMUT_STOP", **self.get_stats())

    async def add_audio(self, audio_chunk: np.ndarray) -> None:
        if not self.is_running:
            return
        values = np.asarray(audio_chunk, dtype=np.float32).reshape(-1)
        if self.sample_rate != MODEL_SAMPLE_RATE:
            pcm16 = np.clip(values * 32768.0, -32768, 32767).astype(np.int16).tobytes()
            converted, self._rate_state = audioop.ratecv(
                pcm16, 2, 1, self.sample_rate, MODEL_SAMPLE_RATE, self._rate_state
            )
            values = (
                np.frombuffer(converted, dtype=np.int16).astype(np.float32) / 32768.0
            )
        await self._append_model_audio(values)

    async def add_audio_bytes(self, ulaw_bytes: bytes) -> None:
        """Accept native PySIP 8 kHz PCMU frames."""
        if not self.is_running:
            return
        self._frames_received += 1
        pcm16 = audioop.ulaw2lin(ulaw_bytes, 2)
        converted, self._rate_state = audioop.ratecv(
            pcm16, 2, 1, 8000, MODEL_SAMPLE_RATE, self._rate_state
        )
        values = np.frombuffer(converted, dtype=np.int16).astype(np.float32) / 32768.0
        await self._append_model_audio(values)

    async def _append_model_audio(self, values: np.ndarray) -> None:
        if not len(values):
            return
        self._model_audio = np.concatenate((self._model_audio, values))
        while len(self._model_audio) >= FRAME_SAMPLES:
            frame = self._model_audio[:FRAME_SAMPLES]
            self._model_audio = self._model_audio[FRAME_SAMPLES:]
            # Preserve Unmute's flush semantics: while delayed-ASR zero frames
            # advance the model, newly arriving microphone frames are not sent.
            # This is a short (~0.5s) post-pause finalization interval.
            if self._flushing:
                continue
            try:
                self._audio_queue.put_nowait(frame.copy())
            except asyncio.QueueFull:
                self._frames_dropped += 1
                _dlog("UMUT_AUDIO_QUEUE_FULL", dropped=self._frames_dropped)

    async def _send(self, message) -> None:
        packed = msgpack.packb(message, use_bin_type=True, use_single_float=True)
        await self.websocket.send(packed)

    async def _sender_loop(self) -> None:
        try:
            while self.is_running:
                frame = await self._audio_queue.get()
                await self._send({"type": "Audio", "pcm": frame.tolist()})
                self._frames_sent += 1
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _dlog("UMUT_SENDER_ERROR", error=repr(exc))
            logger.exception("Umut STT sender failed")
            self.is_running = False

    async def _receiver_loop(self) -> None:
        try:
            async for packet in self.websocket:
                message = msgpack.unpackb(packet)
                kind = message.get("type")
                if kind == "Word":
                    self._handle_word(
                        str(message.get("text", "")), message.get("start_time")
                    )
                elif kind == "Step":
                    await self._handle_step(message)
                elif kind in ("EndWord", "Marker", "Ready"):
                    continue
                elif kind == "Error":
                    raise RuntimeError(message.get("message", "unknown STT error"))
                else:
                    _dlog("UMUT_UNKNOWN_MESSAGE", message=repr(message)[:500])
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self.is_running:
                _dlog("UMUT_RECEIVER_ERROR", error=repr(exc))
                logger.exception("Umut STT receiver failed")
            self.is_running = False

    def _handle_word(self, text: str, start_time) -> None:
        if not text:
            return
        self._words_received += 1
        if not self._turn_started:
            self._turn_started = True
            self._pause.value = 0.0
            self._fire_turn_resumed("word")
            _dlog("UMUT_SPEECH_STARTED", text=repr(text), start_time=start_time)
        self._turn_words.append(text)
        transcript = self._transcript()
        self._emit_partial(
            STTResult(
                text=transcript,
                is_final=False,
                confidence=1.0,
                timestamp=time.time(),
            )
        )
        _dlog("UMUT_WORD", text=repr(text), transcript=repr(transcript))

    async def _handle_step(self, message) -> None:
        self._current_time += FRAME_TIME_SEC
        probabilities = message.get("prs") or []
        if self._steps_to_ignore > 0:
            self._steps_to_ignore -= 1
        elif len(probabilities) > 2:
            previous = self._pause.value
            value = self._pause.update(FRAME_TIME_SEC, float(probabilities[2]))
            if (
                self.vad_interruption
                and self._bot_speaking
                and value < self.speech_threshold
                and self._started_pc is not None
                and time.perf_counter() - self._started_pc > UNINTERRUPTIBLE_BY_VAD_TIME_SEC
                and not self._speech_signal_active
            ):
                self._speech_signal_active = True
                self._fire_turn_resumed("vad")
            elif value >= self.speech_threshold:
                self._speech_signal_active = False

            if (
                self._turn_started
                and not self._flushing
                and value > self.end_threshold
            ):
                await self._begin_flush(value)
            _dlog(
                "UMUT_STEP",
                step=message.get("step_idx"),
                raw=f"{float(probabilities[2]):.4f}",
                pause=f"{value:.4f}",
                previous=f"{previous:.4f}",
                turn_started=self._turn_started,
                flushing=self._flushing,
                bot_speaking=self._bot_speaking,
            )

        if (
            self._flushing
            and self._flush_until is not None
            and self._current_time > self._flush_until
        ):
            self._finish_turn()

    async def _begin_flush(self, pause_value: float) -> None:
        self._flushing = True
        self._pause_events += 1
        self._flush_until = self._current_time + self.delay_sec
        num_frames = int(math.ceil(self.delay_sec / FRAME_TIME_SEC)) + 1
        zero = np.zeros(FRAME_SAMPLES, dtype=np.float32)
        _dlog(
            "UMUT_PAUSE_DETECTED",
            pause=f"{pause_value:.4f}",
            transcript=repr(self._transcript()),
            flush_frames=num_frames,
            current_time=f"{self._current_time:.3f}",
            flush_until=f"{self._flush_until:.3f}",
        )
        for _ in range(num_frames):
            try:
                self._audio_queue.put_nowait(zero)
            except asyncio.QueueFull:
                await self._audio_queue.put(zero)

    def _finish_turn(self) -> None:
        text = self._transcript().strip()
        self._flushing = False
        self._flush_until = None
        if not text:
            self._reset_turn()
            return
        self._utterance_count += 1
        self._last_user_speech_end_pc = time.perf_counter()
        self._last_strong_near_end_pc = self._last_user_speech_end_pc
        _dlog("UMUT_TURN_FINAL", utterance=self._utterance_count, text=repr(text))
        self._emit_final(
            STTResult(
                text=text,
                is_final=True,
                confidence=1.0,
                utterance_num=self._utterance_count,
                timestamp=time.time(),
            )
        )
        self._reset_turn()

    def _reset_turn(self) -> None:
        self._turn_words = []
        self._turn_started = False
        self._speech_signal_active = False
        self._pause.value = 1.0

    def _transcript(self) -> str:
        result = ""
        for word in self._turn_words:
            if result and word and not result[-1].isspace() and not word[0].isspace():
                result += " "
            result += word
        return result

    def _fire_turn_resumed(self, source: str) -> None:
        if source == "word":
            self._interruptions_word += 1
        else:
            self._interruptions_vad += 1
        _dlog("UMUT_TURN_RESUMED", source=source, bot_speaking=self._bot_speaking)
        if self._bot_speaking and self._turn_resumed_callback is not None:
            try:
                self._turn_resumed_callback()
            except Exception as exc:
                _dlog("UMUT_TURN_RESUMED_ERROR", source=source, error=repr(exc))

    def set_turn_resumed_callback(self, callback: Optional[Callable]) -> None:
        self._turn_resumed_callback = callback

    def set_bot_speaking(self, speaking: bool) -> None:
        self._bot_speaking = bool(speaking)
        if not speaking:
            self._speech_signal_active = False
        _dlog("UMUT_BOT_SPEAKING", value=self._bot_speaking)

    def get_stats(self) -> dict:
        return {
            "provider": "umut",
            "is_running": self.is_running,
            "url": self.stt_url,
            "frames_received": self._frames_received,
            "frames_sent": self._frames_sent,
            "frames_dropped": self._frames_dropped,
            "words_received": self._words_received,
            "pause_events": self._pause_events,
            "utterance_count": self._utterance_count,
            "word_interruptions": self._interruptions_word,
            "vad_interruptions": self._interruptions_vad,
            "pause_prediction": self._pause.value,
            "bot_speaking": self._bot_speaking,
            "uptime_seconds": (
                time.perf_counter() - self._started_pc if self._started_pc else 0.0
            ),
        }
