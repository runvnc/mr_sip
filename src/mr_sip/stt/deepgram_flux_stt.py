"""

Deepgram Flux STT Provider

Real-time conversational speech-to-text using Deepgram's Flux model.
Provides ultra-low latency turn detection with eager end-of-turn processing.
"""
import asyncio
import json
import logging
import time
import threading
import numpy as np
import sys
import os
from typing import Optional, Callable
import wave
from datetime import datetime
from deepgram import DeepgramClient
from deepgram.core.events import EventType
from .base_stt import BaseSTTProvider, STTResult

# Check MR_DEBUG env variable
MR_DEBUG = os.environ.get('MR_DEBUG', '').lower() in ('1', 'true', 'yes')

logger = logging.getLogger(__name__)

BLUE_BG_YELLOW_TEXT = '\x1b[44m\x1b[93m'
RESET_COLOR = '\x1b[0m'

def print_deepgram_event(event_type: str, data: dict):
    """Print a formatted event to console and log raw JSON to file."""
    console_message_parts = [event_type]
    for key, value in data.items():
        value_str = str(value)
        if len(value_str) > 150:
            value_str = value_str[:150] + '...'
        console_message_parts.append(f"{key}: '{value_str}'")
    console_message = ' - '.join(console_message_parts)
    logger.info(f'[DEEPGRAM EVENT] {console_message}')
    log_payload = {'timestamp': datetime.utcnow().isoformat(), 'event_type': event_type, **data}
    try:
        # Only write to log file if MR_DEBUG is enabled
        if MR_DEBUG:
            json_string = json.dumps(log_payload, default=str) + '\n'
            with open('/tmp/deepgram_events.log', 'a') as f:
                f.write(json_string)
                f.flush()
    except Exception as e:
        logger.error(f'Failed to write deepgram event to log file: {e}')

class DeepgramFluxSTT(BaseSTTProvider):
    """Deepgram Flux streaming STT provider with conversational turn detection."""

    def __init__(self, api_key: str, sample_rate: int=16000, encoding: str='linear16', language: str='en', model: str='flux-general-en', eager_eot_threshold: float=0.7, eot_threshold: float=0.8, eot_timeout_ms: int=5000, keyterm: list=None):
        """
        Initialize Deepgram Flux STT provider.
        
        Args:
            api_key: Deepgram API key
            sample_rate: Audio sample rate in Hz (default: 16000)
            language: Language code (default: 'en')
            encoding: Audio encoding format - 'linear16' for PCM or 'mulaw' for ulaw
                      (default: 'linear16')
            model: Deepgram model to use (default: 'flux-general-en')
            eager_eot_threshold: Threshold for EagerEndOfTurn events (0.3-0.9, default: 0.7 - BALANCED)
            eot_threshold: Threshold for EndOfTurn events (0.5-0.9, default: 0.8 - BALANCED)
            eot_timeout_ms: Turn timeout in milliseconds (optional)
            keyterm: List of keyterms to boost recognition of specialized terminology
                     (e.g., ["employee", "HR", "manager"])
        """
        super().__init__(sample_rate=sample_rate)
        self.api_key = api_key
        self.language = language
        self.encoding = encoding
        self.model = model
        self.eager_eot_threshold = eager_eot_threshold
        self.eot_threshold = eot_threshold
        self.eot_timeout_ms = eot_timeout_ms
        self.keyterm = keyterm
        self.client: Optional[DeepgramClient] = None
        self.connection = None
        self.listen_thread = None
        self.shutting_down = False
        self.connection_task = None
        self.utterance_count = 0
        self.last_final_text = ''
        self.connection_start_time = None
        self.draft_response_active = False
        self.connection_healthy = False
        self.last_message_time = None
        self.connection_timeout = float(os.getenv('DEEPGRAM_FLUX_MSG_TIMEOUT', '120'))
        self.total_audio_sent = 0
        self.total_eager_eots = 0
        self.total_turn_resumed = 0
        self.total_finals = 0
        self.latencies = []
        self.audio_buffer = []
        self.main_loop = None
        try:
            self.main_loop = asyncio.get_running_loop()
        except RuntimeError:
            self.main_loop = None
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 3
        self.total_reconnections = 0
        self._on_turn_resumed_callback: Optional[Callable[[], None]] = None
        self.debug_wav_file = None

    async def start(self) -> None:
        """Connect to Deepgram Flux API."""
        if self.is_running:
            logger.warning('DeepgramFluxSTT already running')
            return
        self.shutting_down = False
        try:
            logger.info(f'Connecting to Deepgram Flux API (model: {self.model})...')
            self.client = DeepgramClient(api_key=self.api_key)
            # Deepgram SDK 5.0.0 requires all parameters to be strings
            connection_params = {
                'model': self.model,
                'encoding': self.encoding,
                'sample_rate': str(self.sample_rate),
                'eager_eot_threshold': str(self.eager_eot_threshold),
                'eot_threshold': str(self.eot_threshold)
            }
            if self.eot_timeout_ms is not None:
                connection_params['eot_timeout_ms'] = str(self.eot_timeout_ms)
            if self.keyterm is not None:
                # keyterm should be comma-separated string if list
                connection_params['keyterm'] = ','.join(self.keyterm) if isinstance(self.keyterm, list) else str(self.keyterm)
            self.is_running = True
            self.connection_start_time = time.time()
            self.connection_task = asyncio.create_task(self._run_connection(**connection_params))
            self.health_monitor_task = asyncio.create_task(self._monitor_connection_health())
            await asyncio.sleep(0.5)
            logger.info('Connected to Deepgram Flux API')
        except Exception as e:
            logger.error(f'Failed to connect to Deepgram Flux: {e}')
            raise

    async def _monitor_connection_health(self):
        """Monitor connection health; do not kill the process, and defer checks until SIP call is established and audio is flowing."""
        await asyncio.sleep(5.0)
        while self.is_running:
            try:
                if not getattr(self, 'sip_call_established', False):
                    await asyncio.sleep(2.0)
                    continue
                if not self.connection_healthy and self.connection_start_time:
                    elapsed = time.time() - self.connection_start_time
                    establish_timeout = float(os.getenv('DEEPGRAM_FLUX_CONNECT_TIMEOUT', '60'))
                    if elapsed > establish_timeout:
                        logger.warning(f'Deepgram Flux connection not healthy after {elapsed:.1f}s; stopping STT and will rely on reconnection logic later')
                        await self.stop()
                        return
                if self.connection_healthy and self.last_message_time:
                    elapsed = time.time() - self.last_message_time
                    if elapsed > self.connection_timeout:
                        logger.warning(f'No messages from Deepgram Flux for {elapsed:.1f} seconds; stopping STT and allowing caller to restart when audio resumes')
                        await self.stop()
                        return
                await asyncio.sleep(5.0)
            except Exception as e:
                logger.error(f'Error in connection health monitor: {e}')
                await asyncio.sleep(1.0)

    async def _run_connection(self, **connection_params):
        """Run the Deepgram connection in synchronous context manager."""
        logger.info(f'Starting Deepgram connection with params: {connection_params}')
        try:
            with self.client.listen.v2.connect(**connection_params) as connection:
                self.connection = connection
                connection.on(EventType.OPEN, self._on_open)
                connection.on(EventType.MESSAGE, self._on_message)
                connection.on(EventType.ERROR, self._on_error)
                connection.on(EventType.CLOSE, self._on_close)
                self.listen_thread = threading.Thread(target=connection.start_listening, daemon=True)
                self.listen_thread.start()
                logger.info(f'Started Deepgram listen thread: {self.listen_thread.name}')
                logger.info('Entering connection keep-alive loop')
                while self.is_running:
                    await asyncio.sleep(0.1)
                logger.info('Exiting connection keep-alive loop')
        except Exception as e:
            logger.error(f'Error in Deepgram connection: {e}')
            self.is_running = False
            self._exit_process(f'Connection error: {e}')

    def _exit_process(self, reason: str):
        """Deprecated: do not exit the process from the STT layer."""
        logger.warning(f'Deepgram Flux connection issue (no sys.exit): {reason}')

    async def stop(self) -> None:
        """Disconnect from Deepgram Flux API with comprehensive cleanup."""
        if not self.is_running:
            logger.debug('DeepgramFluxSTT already stopped')
            return
        logger.info('Stopping Deepgram Flux STT provider...')
        self.shutting_down = True
        self.is_running = False
        try:
            if self.connection:
                try:
                    logger.info('Closing Deepgram connection...')
                    if hasattr(self.connection, 'finish'):
                        self.connection.finish()
                    elif hasattr(self.connection, 'close'):
                        self.connection.close()
                except Exception as e:
                    logger.error(f'Error closing connection: {e}')
            if self.listen_thread and self.listen_thread.is_alive():
                try:
                    logger.info('Waiting for listen thread to stop...')
                    self.listen_thread.join(timeout=2.0)
                    if self.listen_thread.is_alive():
                        logger.warning('Listen thread did not stop in time')
                except Exception as e:
                    logger.error(f'Error joining listen thread: {e}')
            if self.connection_task and (not self.connection_task.done()):
                try:
                    self.connection_task.cancel()
                    logger.debug('Connection task cancelled')
                except Exception as e:
                    logger.error(f'Error cancelling connection task: {e}')
            if hasattr(self, 'health_monitor_task') and (not self.health_monitor_task.done()):
                try:
                    self.health_monitor_task.cancel()
                    logger.debug('Health monitor task cancelled')
                except Exception as e:
                    logger.error(f'Error cancelling health monitor: {e}')
            self.connection = None
            self.client = None
            self.listen_thread = None
            # Only clear numpy audio buffer, not raw bytes buffer
            if hasattr(self, 'audio_buffer'):
                self.audio_buffer.clear()
            self.raw_audio_buffer = []
            logger.info('Deepgram Flux STT provider stopped and cleaned up')
        except Exception as e:
            logger.error(f'Error during stop cleanup: {e}')

    async def add_audio(self, audio_chunk: np.ndarray) -> None:
        """Send audio chunk to Deepgram Flux."""
        self.audio_buffer.append(audio_chunk.copy())
        if len(self.audio_buffer) > 10:
            self.audio_buffer.pop(0)
        if not self.is_running or not self.connection:
            logger.debug('Cannot send audio: not connected to Deepgram Flux')
            logger.warning('🔄 TRIGGERING BUFFERED RECONNECTION - Connection lost!')
            self._schedule_coroutine_threadsafe(self._reconnect_with_buffer())
            return
        try:
            if audio_chunk.dtype != np.float32:
                audio_chunk = audio_chunk.astype(np.float32)
            if audio_chunk.size:
                audio_chunk = audio_chunk - float(audio_chunk.mean())
            audio_chunk = np.clip(audio_chunk, -1.0, 1.0)
            audio_int16 = (audio_chunk * 32767.0).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            self.connection.send_media(audio_bytes)
            self.total_audio_sent += len(audio_bytes)
            logger.debug(f'Sent {len(audio_bytes)} bytes to Deepgram Flux (total: {self.total_audio_sent})')
        except Exception as e:
            error_msg = str(e)
            if 'sent' in error_msg and 'received' in error_msg and ('OK' in error_msg):
                logger.debug(f'WebSocket status (not an error): {e}')
            else:
                logger.error(f'Error sending audio to Deepgram Flux: {e}')
                logger.warning('🔄 TRIGGERING BUFFERED RECONNECTION after send error!')
                self._schedule_coroutine_threadsafe(self._reconnect_with_buffer())

    async def add_audio_bytes(self, audio_bytes: bytes) -> None:
        """Send raw audio bytes to Deepgram Flux.
        
        This method is optimized for mulaw encoding where we receive raw bytes
        directly from PySIP and can pass them through without conversion.
        
        Args:
            audio_bytes: Raw audio bytes (mulaw or PCM depending on encoding setting)
        """
        # Buffer for reconnection
        if not hasattr(self, 'raw_audio_buffer'):
            self.raw_audio_buffer = []
        self.raw_audio_buffer.append(audio_bytes)
        if len(self.raw_audio_buffer) > 20:  # ~400ms at 20ms frames
            self.raw_audio_buffer.pop(0)
        
        if not self.is_running or not self.connection:
            logger.debug('Cannot send audio: not connected to Deepgram Flux')
            logger.warning('🔄 TRIGGERING BUFFERED RECONNECTION - Connection lost!')
            self._schedule_coroutine_threadsafe(self._reconnect_with_buffer())
            return
        
        try:
            self.connection.send_media(audio_bytes)
            self.total_audio_sent += len(audio_bytes)
            logger.debug(f'Sent {len(audio_bytes)} raw bytes to Deepgram Flux (total: {self.total_audio_sent})')
        except Exception as e:
            logger.error(f'Error sending raw audio to Deepgram Flux: {e}')
            logger.warning('🔄 TRIGGERING BUFFERED RECONNECTION after send error!')
            self._schedule_coroutine_threadsafe(self._reconnect_with_buffer())

    def _on_open(self, *args) -> None:
        """Handle connection open event."""
        print_deepgram_event('ConnectionStatus', {'status': 'OPENED', 'message': 'ready to receive audio'})
        logger.info('Deepgram Flux connection opened - ready to receive audio')
        self.connection_healthy = True
        logger.debug('Deepgram Flux connection opened')

    def _on_close(self, *args) -> None:
        """Handle connection close event."""
        print_deepgram_event('ConnectionStatus', {'status': 'CLOSED', 'args': args})
        logger.info('Deepgram Flux connection closed')
        if self.shutting_down:
            logger.info('Ignoring close event during shutdown')
            return
        if self.is_running:
            self.connection_healthy = False
            print_deepgram_event('ReconnectionTrigger', {'reason': 'unexpected close'})
            logger.warning('\ud83d\udd04 TRIGGERING BUFFERED RECONNECTION due to unexpected close')
            logger.warning('🔄 TRIGGERING BUFFERED RECONNECTION due to unexpected close')
            self._schedule_coroutine_threadsafe(self._reconnect_with_buffer())

    def _on_error(self, error) -> None:
        """Handle connection error event."""
        error_str = str(error)
        logger.error(f'Deepgram Flux connection error: {error}')
        print_deepgram_event('ConnectionError', {'error': error_str})
        if self.shutting_down:
            logger.info('Ignoring error event during shutdown')
            return
        if '1011' in error_str or 'policy violation' in error_str.lower() or 'connection' in error_str.lower():
            logger.warning(f'🔄 TRIGGERING BUFFERED RECONNECTION due to error: {error}')
            self._schedule_coroutine_threadsafe(self._reconnect_with_buffer())

    def _on_message(self, message) -> None:
        """Handle incoming message from Deepgram Flux."""
        msg_attrs = {}
        for attr in dir(message):
            if not attr.startswith('_'):
                try:
                    value = getattr(message, attr)
                    if not callable(value):
                        msg_attrs[attr] = value
                except:
                    pass
        print_deepgram_event('MessageReceived', msg_attrs)
        msg_type = getattr(message, 'type', 'unknown')
        logger.info(f'DEEPGRAM DEBUG Received message from Deepgram Flux: {msg_type}')
        self.last_message_time = time.time()
        try:
            if not hasattr(message, 'type') or message.type != 'TurnInfo':
                logger.debug(f"Received non-TurnInfo message: {getattr(message, 'type', 'unknown')}")
                return
            event = getattr(message, 'event', None)
            transcript = getattr(message, 'transcript', '').strip()
            if not transcript:
                logger.debug(f'Received TurnInfo message with empty transcript, event: {event}')
                return
            latency = time.time() - self.connection_start_time if self.connection_start_time else 0
            self.latencies.append(latency)
            if event == 'EagerEndOfTurn':
                logger.debug('DEEPGRAM DEBUG EagerEndOfTurn')
                self._handle_eager_eot(transcript, latency)
            elif event == 'TurnResumed' or event == 'StartOfTurn':
                logger.debug('DEEPGRAM DEBUG turnresumed for turnresumed or startofturn')
                self._handle_turn_resumed(transcript, latency, event)
            elif event == 'EndOfTurn':
                logger.debug('DEEPGRAM DEBUG end of turn')
                self._handle_end_of_turn(transcript, latency)
            else:
                logger.debug(f'DEEPGRAM DEBUG Unknown Flux event: {event}')
        except Exception as e:
            logger.error(f'Error handling Flux message: {e}')

    def _handle_eager_eot(self, transcript: str, latency: float) -> None:
        """Handle EagerEndOfTurn event - start preparing response."""
        self.total_eager_eots += 1
        self.draft_response_active = True
        print_deepgram_event('EagerEOT', {'transcript': transcript, 'latency_ms': f'{latency * 1000:.0f}'})
        logger.info(f'[EAGER EOT] {transcript} (latency: {latency * 1000:.0f}ms)')
        result = STTResult(text=transcript, is_final=False, is_eager_eot=True, confidence=0.8, timestamp=time.time())
        self._emit_partial(result)

    def set_turn_resumed_callback(self, callback: Optional[Callable[[], None]]) -> None:
        """Set callback for TurnResumed events."""
        self._on_turn_resumed_callback = callback

    def _handle_turn_resumed(self, transcript: str, latency: float, event_name: str='TurnResumed') -> None:
        """Handle user interruption (TurnResumed or StartOfTurn) - cancel draft response."""
        self.total_turn_resumed += 1
        self.draft_response_active = False
        print_deepgram_event(event_name, {'message': 'User interruption detected', 'latency_ms': f'{latency * 1000:.0f}'})
        logger.info(f'[{event_name.upper()}] User interruption detected (latency: {latency * 1000:.0f}ms)')
        if self._on_turn_resumed_callback:
            try:
                self._on_turn_resumed_callback()
            except Exception as e:
                logger.error(f'Error in turn resumed callback: {e}')

    def _handle_end_of_turn(self, transcript: str, latency: float) -> None:
        """Handle EndOfTurn event - finalize response."""
        self.utterance_count += 1
        self.total_finals += 1
        self.last_final_text = transcript
        self.draft_response_active = False
        print_deepgram_event('EndOfTurn', {'utterance_num': self.utterance_count, 'transcript': transcript, 'latency_ms': f'{latency * 1000:.0f}'})
        logger.info(f'[FINAL #{self.utterance_count}] {transcript} (latency: {latency * 1000:.0f}ms)')
        result = STTResult(text=transcript, is_final=True, is_eager_eot=False, confidence=0.95, timestamp=time.time())
        result.utterance_num = self.utterance_count
        self._emit_final(result)

    def _should_attempt_reconnect(self) -> bool:
        """Check if we should attempt reconnection."""
        if self.shutting_down:
            logger.info('Skipping reconnection - shutting down')
            return False
        if not hasattr(self, 'sip_call_established') or not getattr(self, 'sip_call_established', False):
            logger.warning('🔍 DEBUG: Skipping reconnection - SIP call not established yet')
            return False
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.warning(f'Max reconnection attempts reached: {self.reconnect_attempts}/{self.max_reconnect_attempts}')
            return False
        return True

    def set_sip_call_established(self, established: bool):
        """Set whether the SIP call is established (prevents premature reconnection)."""
        self.sip_call_established = established
        logger.error(f'🔍 DEBUG: SIP call established status set to: {established}')

    async def _reconnect_with_buffer(self):
        """Reconnect and immediately send buffered audio."""
        if not self._should_attempt_reconnect():
            return
        try:
            self.reconnect_attempts += 1
            self.total_reconnections += 1
            logger.error('=' * 80)
            logger.error(f'BUFFERED RECONNECTION #{self.total_reconnections} (attempt {self.reconnect_attempts}/{self.max_reconnect_attempts})')
            if len(self.audio_buffer) == 0:
                backoff_times = [1.0, 2.0, 5.0]
                backoff_time = backoff_times[min(self.reconnect_attempts - 1, len(backoff_times) - 1)]
                logger.error(f'NO AUDIO BUFFERED - Using slow backoff: {backoff_time}s')
            else:
                backoff_times = [0.1, 0.2, 0.5]
                backoff_time = backoff_times[min(self.reconnect_attempts - 1, len(backoff_times) - 1)]
                logger.error(f'AUDIO BUFFERED ({len(self.audio_buffer)} chunks) - Using fast backoff: {backoff_time}s')
            logger.error('=' * 80)
            if backoff_time > 0:
                logger.warning(f'Waiting {backoff_time}s before reconnection...')
                await asyncio.sleep(backoff_time)
            logger.warning(f'Starting reconnection to Deepgram Flux...')
            await self.stop()
            if not getattr(self, 'sip_call_established', False):
                logger.warning('Skipping Deepgram reconnect: SIP call not yet established')
                return
            await self.start()
            await asyncio.sleep(0.2)
            if len(self.audio_buffer) > 0:
                logger.warning(f'Sending {len(self.audio_buffer)} buffered audio chunks immediately...')
                for i, chunk in enumerate(self.audio_buffer):
                    if self.connection and self.connection_healthy:
                        await self._send_audio_chunk_direct(chunk)
                        await asyncio.sleep(0.01)
                    else:
                        logger.error(f'Connection lost during buffer send at chunk {i}')
                        break
            # Also send raw audio buffer if using mulaw
            if hasattr(self, 'raw_audio_buffer') and len(self.raw_audio_buffer) > 0:
                logger.warning(f'Sending {len(self.raw_audio_buffer)} buffered raw audio chunks...')
                for i, chunk in enumerate(self.raw_audio_buffer):
                    if self.connection and self.connection_healthy:
                        self.connection.send_media(chunk)
                        await asyncio.sleep(0.01)
                    else:
                        logger.error(f'Connection lost during raw buffer send at chunk {i}')
                        break
            else:
                logger.warning('No buffered audio - connection ready for new audio')
            logger.warning(f'RECONNECTION COMPLETE - resuming normal operation')
            self.reconnect_attempts = 0
        except Exception as e:
            logger.error(f'RECONNECTION FAILED: {e}')

    async def _send_audio_chunk_direct(self, audio_chunk: np.ndarray) -> None:
        """Send audio chunk directly to Deepgram (for buffered sending)."""
        try:
            if audio_chunk.dtype != np.float32:
                audio_chunk = audio_chunk.astype(np.float32)
            if audio_chunk.size:
                audio_chunk = audio_chunk - float(audio_chunk.mean())
            audio_chunk = np.clip(audio_chunk, -1.0, 1.0)
            audio_int16 = (audio_chunk * 32767.0).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            self.connection.send_media(audio_bytes)
        except Exception as e:
            logger.error(f'Error in direct audio send: {e}')

    async def _reconnect(self) -> None:
        """Attempt to reconnect to Deepgram Flux."""
        if not self.is_running:
            return
        try:
            await asyncio.sleep(1.0)
            await self.stop()
            await self.start()
        except Exception as e:
            logger.error(f'Failed to reconnect to Deepgram Flux: {e}')

    def get_stats(self) -> dict:
        """Get statistics about the Deepgram Flux connection."""
        avg_latency = sum(self.latencies) / len(self.latencies) if self.latencies else 0
        return {'provider': 'deepgram-flux', 'model': self.model, 'is_running': self.is_running, 'utterance_count': self.utterance_count, 'total_audio_sent_bytes': self.total_audio_sent, 'total_eager_eots': self.total_eager_eots, 'total_turn_resumed': self.total_turn_resumed, 'total_finals': self.total_finals, 'average_latency_ms': avg_latency * 1000, 'connection_duration_seconds': time.time() - self.connection_start_time if self.connection_start_time else 0, 'eager_eot_threshold': self.eager_eot_threshold, 'eot_threshold': self.eot_threshold, 'draft_response_active': self.draft_response_active, 'total_reconnections': self.total_reconnections, 'current_reconnect_attempts': self.reconnect_attempts, 'audio_buffer_size': len(self.audio_buffer)}

    def _schedule_coroutine_threadsafe(self, coro):
        """Schedule a coroutine to run in the main event loop from any thread."""
        if self.main_loop and (not self.main_loop.is_closed()):
            try:
                future = asyncio.run_coroutine_threadsafe(coro, self.main_loop)
                return future
            except Exception as e:
                logger.error(f'Failed to schedule coroutine from thread: {e}')
        else:
            logger.warning('No main event loop available for cross-thread scheduling')
        return None