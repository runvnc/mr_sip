"""
SIP Client for MindRoot - Version 2 with STT Provider Interface

This version uses the abstract STT provider interface, allowing easy switching
between Deepgram, Whisper, and future STT backends.

Key improvements over v1:
- Abstract STT provider interface
- inotify-based audio capture (eliminates polling)
- Support for partial transcription results
- Cleaner separation of concerns
"""
import os
import sys
import time
import struct
import threading
import numpy as np
from scipy import signal
from queue import Empty
from baresipy import BareSIP
from datetime import datetime
from pathlib import Path
import asyncio
import logging
from typing import Callable, Optional, Any
from .audio_handler import AudioHandler
from .stt import create_stt_provider, BaseSTTProvider, STTResult
from .audio import InotifyAudioCapture
from .audio import JACKAudioCapture
from .audio_combiner import AudioCombiner
from lib.providers.services import service_manager
logging.getLogger('inotify.adapters').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

class MindRootSIPBotV2(BareSIP):
    """SIP phone bot integrated with MindRoot's AI agent system (V2)."""

    def __init__(self, user, password, gateway, audio_dir='.', on_utterance_callback=None, stt_provider: str=None, stt_config: dict=None, context=None):
        """
        Args:
            user: SIP username
            password: SIP password
            gateway: SIP gateway
            audio_dir: Directory where baresip saves recordings
            on_utterance_callback: Async function called with each complete utterance
                                  Signature: async callback(text, utterance_num, timestamp, context)
            stt_provider: STT provider name ('deepgram', 'whisper_vad', etc.)
                         If None, uses STT_PROVIDER env var or defaults to 'deepgram_flux'
            stt_config: Additional configuration for STT provider
            context: MindRoot ChatContext
        """
        self.audio_dir = audio_dir or os.path.expanduser('~/.baresip')
        super().__init__(user, password, gateway, block=False)
        self.context = context
        self.on_utterance_callback = on_utterance_callback
        self.call_start_time = None
        self.current_dec_file = None
        self.current_enc_file = None
        self.stt_provider_name = stt_provider or os.getenv('STT_PROVIDER', 'deepgram_flux')
        self.stt_config = stt_config or {}
        self.stt: Optional[BaseSTTProvider] = None
        self.audio_capture = None
        self.audio_capture_method = os.getenv('AUDIO_CAPTURE_METHOD', 'jack').strip().lower()
        if self.audio_capture_method == 'jack' and JACKAudioCapture is None:
            logger.critical('JACK input capture is required but unavailable. Install python-jack-client (pip install jack-client), ensure JACK server is running, and set AUDIO_CAPTURE_METHOD=inotify only if you explicitly want file capture.')
            sys.exit(2)
        self.utterances = []
        self.last_partial_text = ''
        self.active_ai_task_id = None
        self.draft_response_active = False
        self.last_eager_eot_text = ''
        self.audio_handler = AudioHandler()
        self.tts_audio_queue = None
        self.tts_sender_task = None
        self.audio_combiner = AudioCombiner()
        try:
            self.main_loop = asyncio.get_running_loop()
        except RuntimeError:
            self.main_loop = None

    def __del__(self):
        """Destructor to ensure STT and audio capture are stopped."""
        logger.info(f"MindRootSIPBotV2 instance for context {(self.context.log_id if self.context else 'N/A')} is being destroyed.")
        if self.stt and self.stt.is_running:
            logger.warning('STT provider was still running. Forcing stop.')
            self._schedule_coroutine(self.stt.stop())
        if self.audio_capture and self.audio_capture.is_running:
            logger.warning('Audio capture was still running. Forcing stop.')
            self._schedule_coroutine(self.audio_capture.stop())

    def _schedule_coroutine(self, coro):
        """Schedule a coroutine to run in the main event loop from any thread."""
        logger.error(f'🔍 DEBUG: _schedule_coroutine called with: {coro}')
        if self.main_loop and (not self.main_loop.is_closed()):
            try:
                future = asyncio.run_coroutine_threadsafe(coro, self.main_loop)
                logger.error(f'🔍 DEBUG: Coroutine scheduled successfully')
                return future
            except Exception as e:
                logger.error(f'🔍 DEBUG: Failed to schedule coroutine: {e}')
                logger.error(f'Failed to schedule coroutine: {e}')
        else:
            logger.error(f'🔍 DEBUG: No main loop available for scheduling')
        return None

    def handle_call_established(self):
        """When call connects, setup JACK and start transcription."""
        logger.error('🔍 DEBUG: handle_call_established() CALLED - This is when STT should start')
        logger.info('=== CALL ESTABLISHED ===')
        self.call_start_time = datetime.now()
        if not self.audio_handler.jack_enabled:
            self.audio_handler.setup_jack_audio()
            time.sleep(0.1)
        self.audio_handler.configure_baresip_jack(self)
        self.audio_handler.connect_jack_to_baresip()
        if self.audio_capture_method == 'jack' and JACKAudioCapture is not None:
            self._schedule_coroutine(self._setup_stt_and_capture_jack())
        else:
            self._schedule_coroutine(self._setup_stt_and_capture())
        if self.main_loop and (not self.main_loop.is_closed()):
            try:
                if self.tts_audio_queue is None:
                    future = asyncio.run_coroutine_threadsafe(self._create_tts_queue(), self.main_loop)
                    future.result(timeout=5.0)
                future = asyncio.run_coroutine_threadsafe(self._start_tts_sender(), self.main_loop)
                logger.info('TTS audio sender scheduled')
            except Exception as e:
                logger.error(f'Failed to start TTS audio sender: {e}')

    async def _setup_stt_and_capture(self):
        """Setup STT provider and audio capture with pre-buffering."""
        logger.error('🔍 DEBUG: _setup_stt_and_capture() STARTED')
        logger.error(f'🔍 DEBUG: Current STT provider exists: {self.stt is not None}')
        try:
            await self._find_current_audio_files()
            if not self.current_dec_file:
                logger.error('Could not find audio recording file')
                return
            logger.info(f'Using audio file: {self.current_dec_file}')
            logger.info('Starting audio capture to pre-buffer audio...')
            self.audio_capture = InotifyAudioCapture(audio_file=self.current_dec_file, target_sample_rate=16000, chunk_callback=self._on_audio_chunk_prebuffer)
            await self.audio_capture.start()
            logger.info('Audio capture started, waiting for audio data...')
            max_wait_time = 10.0
            wait_start = time.time()
            logger.error(f"🔍 DEBUG: Waiting for pre-buffer, current size: {len(getattr(self, 'audio_prebuffer', []))}")
            while len(getattr(self, 'audio_prebuffer', [])) < 2:
                if time.time() - wait_start > max_wait_time:
                    logger.error('Timeout waiting for audio data')
                    return
                await asyncio.sleep(0.1)
            logger.info(f'Got {len(self.audio_prebuffer)} audio chunks, now starting STT provider...')
            logger.error(f'🔍 DEBUG: About to create STT provider: {self.stt_provider_name}')
            logger.info(f'Creating STT provider: {self.stt_provider_name}')
            self.stt = create_stt_provider(self.stt_provider_name, **self.stt_config)
            logger.error(f'🔍 DEBUG: STT provider created successfully: {self.stt}')
            self.audio_capture.target_sample_rate = self.stt.sample_rate
            self.stt.set_callbacks(on_partial=self._on_partial_result, on_final=self._on_final_result)
            logger.error(f'🔍 DEBUG: STT callbacks set')
            if hasattr(self.stt, 'set_turn_resumed_callback'):
                self.stt.set_turn_resumed_callback(self._handle_turn_resumed)
                logger.info('Barge-in detection enabled (clears TTS queue only)')
            if hasattr(self.stt, 'main_loop'):
                self.stt.main_loop = self.main_loop
            if hasattr(self.stt, 'set_sip_call_established'):
                self.stt.set_sip_call_established(True)
            logger.error('🔍 DEBUG: About to start STT provider (this opens Deepgram connection)')
            try:
                await self.stt.start()
            except Exception as e:
                logger.error(f'🔥 CRITICAL: STT start() failed with exception: {e}')
                import traceback
                logger.error(f'🔥 Traceback: {traceback.format_exc()}')
                raise
            logger.info(f'STT provider started: {self.stt_provider_name}')
            logger.error(f'🔄 DEBUG: Switching audio callback from pre-buffer to normal mode')
            self.audio_capture.chunk_callback = self._on_audio_chunk
            logger.error(f'🔄 DEBUG: Callback switched successfully, callback is now: {self.audio_capture.chunk_callback.__name__}')
            logger.error(f'🔄 DEBUG: Audio capture callback verification: {self.audio_capture.chunk_callback == self._on_audio_chunk}')
            logger.error(f'🔍 DEBUG: Sending {len(self.audio_prebuffer)} pre-buffered chunks immediately')
            logger.info(f'Sending {len(self.audio_prebuffer)} pre-buffered chunks to STT...')
            for chunk in self.audio_prebuffer:
                await self.stt.add_audio(chunk)
                await asyncio.sleep(0.01)
            logger.info('STT setup complete - audio flowing immediately')
        except Exception as e:
            logger.error(f'🔍 DEBUG: Exception in _setup_stt_and_capture: {e}')
            logger.error(f'Error setting up STT and capture: {e}')
            import traceback
            logger.critical(f'FATAL: STT setup failed, cannot continue without Deepgram')
            logger.critical(traceback.format_exc())
            import sys
            sys.exit(1)

    async def _setup_stt_and_capture_jack(self):
        """Setup STT and JACK-based audio capture (no file I/O)."""
        logger.error('🔍 DEBUG: _setup_stt_and_capture_jack() STARTED')
        try:
            if JACKAudioCapture is None:
                logger.critical('JACKAudioCapture not available; exiting due to no-fallback policy')
                sys.exit(2)
            logger.info('Starting JACK audio capture to pre-buffer audio...')
            self.audio_capture = JACKAudioCapture(target_sample_rate=16000, chunk_duration_s=0.25, chunk_callback=self._on_audio_chunk_prebuffer, stereo_mix=True, agc_target_rms=0.15, agc_max_gain=20.0)
            await self.audio_capture.start()
            max_wait_time = 10.0
            wait_start = time.time()
            if not hasattr(self, 'audio_prebuffer'):
                self.audio_prebuffer = []
            while len(self.audio_prebuffer) < 2:
                if time.time() - wait_start > max_wait_time:
                    logger.critical('Timeout waiting for JACK audio data; exiting due to no-fallback policy')
                    sys.exit(2)
                await asyncio.sleep(0.1)
            logger.info(f'Got {len(self.audio_prebuffer)} JACK audio chunks, now starting STT provider...')
            logger.error(f'🔍 DEBUG: About to create STT provider (JACK path): {self.stt_provider_name}')
            self.stt = create_stt_provider(self.stt_provider_name, **self.stt_config)
            self.audio_capture.target_sample_rate = self.stt.sample_rate
            self.stt.set_callbacks(on_partial=self._on_partial_result, on_final=self._on_final_result)
            if hasattr(self.stt, 'set_turn_resumed_callback'):
                self.stt.set_turn_resumed_callback(self._handle_turn_resumed)
                logger.info('Barge-in detection enabled (clears TTS queue only)')
            if hasattr(self.stt, 'main_loop'):
                self.stt.main_loop = self.main_loop
            if hasattr(self.stt, 'set_sip_call_established'):
                self.stt.set_sip_call_established(True)
            logger.error('🔍 DEBUG: About to start STT provider (JACK path)')
            await self.stt.start()
            logger.info(f'STT provider started: {self.stt_provider_name}')
            logger.error(f'🔄 DEBUG (JACK): Switching audio callback from pre-buffer to normal mode')
            self.audio_capture.chunk_callback = self._on_audio_chunk
            logger.error(f'🔄 DEBUG (JACK): Callback switched, callback is now: {self.audio_capture.chunk_callback.__name__}')
            logger.error(f'🔄 DEBUG (JACK): Verification: {self.audio_capture.chunk_callback == self._on_audio_chunk}')
            logger.info(f'Sending {len(self.audio_prebuffer)} pre-buffered JACK chunks to STT...')
            for chunk in self.audio_prebuffer:
                await self.stt.add_audio(chunk)
                await asyncio.sleep(0.01)
            logger.info('STT setup complete (JACK) - audio flowing immediately')
        except Exception as e:
            logger.error(f'Error setting up STT and JACK capture: {e}')

    async def _on_audio_chunk_prebuffer(self, audio_chunk: np.ndarray):
        """Callback for audio chunks during pre-buffering phase."""
        if not hasattr(self, 'audio_prebuffer'):
            self.audio_prebuffer = []
        self.audio_prebuffer.append(audio_chunk.copy())
        if len(self.audio_prebuffer) > 10:
            self.audio_prebuffer.pop(0)
        logger.debug(f'Pre-buffered audio chunk, total: {len(self.audio_prebuffer)} chunks')

    async def _on_audio_chunk(self, audio_chunk: np.ndarray):
        """Callback for audio chunks from capture."""
        if self.stt and self.stt.is_running:
            if not hasattr(self, '_audio_chunk_count'):
                self._audio_chunk_count = 0
            self._audio_chunk_count += 1
            if self._audio_chunk_count % 100 == 0:
                logger.info(f'Audio flowing to STT: {self._audio_chunk_count} chunks sent, STT running: {self.stt.is_running}')
            logger.debug(f'Sending audio chunk of size {len(audio_chunk)} to STT')
            try:
                await self.stt.add_audio(audio_chunk)
            except Exception as e:
                logger.error(f'Error feeding audio to STT: {e}')

    def _on_partial_result(self, result: STTResult):
        """Callback for partial transcription results."""
        if result.text != self.last_partial_text:
            logger.debug(f'[PARTIAL] {result.text} (confidence: {result.confidence:.2f}, eager_eot: {result.is_eager_eot})')
            logger.info(f'[PARTIAL] {result.text} (confidence: {result.confidence:.2f}, eager_eot: {result.is_eager_eot})')
            self.last_partial_text = result.text
            if hasattr(result, 'is_eager_eot') and result.is_eager_eot:
                logger.info(f'[EAGER EOT] Starting AI response preparation for: {result.text}')
                self.draft_response_active = True
                self.last_eager_eot_text = result.text
                if self.on_utterance_callback:
                    utterance_num = len(self.utterances) + 1
                    self._schedule_coroutine(self._call_utterance_callback(result.text, utterance_num, result.timestamp or time.time(), is_eager=True))

    def _on_final_result(self, result: STTResult):
        """Callback for final transcription results."""
        utterance_data = {'number': result.utterance_num or len(self.utterances) + 1, 'text': result.text, 'timestamp': result.timestamp or time.time(), 'confidence': result.confidence, 'time_str': time.strftime('%H:%M:%S', time.localtime(result.timestamp or time.time()))}
        from .sip_manager import get_session_manager
        session_manager = get_session_manager()

        async def unset_halt_flag():
            logger.debug(f'UNSET HALT AUDIO! text is {result.text}')
            session = await session_manager.get_session(self.context.log_id)
            if session:
                session.halt_audio_out = False
        self._schedule_coroutine(unset_halt_flag())
        self.utterances.append(utterance_data)
        logger.info(f"[{utterance_data['time_str']}] Utterance #{utterance_data['number']}: {result.text} (confidence: {result.confidence:.2f})")
        should_send = not (result.text == self.last_eager_eot_text and self.last_eager_eot_text)
        if result.text == self.last_eager_eot_text and self.last_eager_eot_text:
            logger.info(f"[FINAL] Skipping duplicate - already sent via EagerEOT: '{result.text}'")
        else:
            logger.info(f"[FINAL] Sending to agent: '{result.text}'")
            if self.on_utterance_callback:
                self._schedule_coroutine(self._call_utterance_callback(result.text, utterance_data['number'], utterance_data['timestamp'], is_eager=False))
        self.draft_response_active = False
        self.last_eager_eot_text = ''
        self.active_ai_task_id = None
        self.last_partial_text = ''

    async def _call_utterance_callback(self, text: str, utterance_num: int, timestamp: float, is_eager: bool=False):
        """Call the utterance callback."""
        try:
            if asyncio.iscoroutinefunction(self.on_utterance_callback):
                await self.on_utterance_callback(text, utterance_num, timestamp, self.context, is_eager=is_eager)
            else:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self.on_utterance_callback, text, utterance_num, timestamp, self.context)
        except Exception as e:
            logger.error(f'Error in utterance callback: {e}')

    async def _cancel_ai_response(self):
        """Cancel active AI response using service manager."""
        if not self.context or not self.context.log_id:
            logger.warning('Cannot cancel AI response: no context or log_id')
            return
        try:
            result = await service_manager.cancel_active_response(log_id=self.context.log_id, context=self.context)
            logger.info(f'AI response cancelled: {result}')
            self.draft_response_active = False
            self.active_ai_task_id = None
        except Exception as e:
            logger.error(f'Error cancelling AI response: {e}')

    def _stop_tts_immediately(self):
        """Clear TTS queue on barge-in - don't touch JACK output."""
        try:
            if self.tts_audio_queue is not None:
                try:
                    while not self.tts_audio_queue.empty():
                        _ = self.tts_audio_queue.get_nowait()
                    logger.info('TTS queue flushed on barge-in')
                except Exception as e:
                    logger.debug(f'TTS queue drain exception (non-fatal): {e}')
        except Exception as e:
            logger.error(f'_stop_tts_immediately encountered an error: {e}')

    def _handle_turn_resumed(self):
        """Handle TurnResumed event from Deepgram Flux."""
        if self.context and self.context.log_id:
            from .sip_manager import get_session_manager
            session_manager = get_session_manager()

            async def set_halt_flag():
                session = await session_manager.get_session(self.context.log_id)
                if session:
                    session.halt_audio_out = True
                    logger.info('[BARGE-IN] User speaking (StartOfTurn/TurnResumed) - halting TTS output')
            self._schedule_coroutine(set_halt_flag())
        self.last_eager_eot_text = ''
        if self.draft_response_active:
            logger.info(f'[TURN RESUMED] Cancelling draft AI response')
            self._schedule_coroutine(self._cancel_ai_response())
        else:
            logger.debug(f'[TURN RESUMED] No active draft response to cancel')

    async def _find_current_audio_files(self):
        """Find the audio files that baresip created for this call."""
        logger.info('Polling for audio files...')
        max_attempts = 50
        attempt = 0
        while attempt < max_attempts:
            dump_files = [f for f in os.listdir(os.getcwd()) if f.startswith('dump-') and f.endswith('.wav')]
            if dump_files:
                logger.info(f'Found audio files after {attempt * 0.1:.1f}s')
                break
            await asyncio.sleep(0.1)
            attempt += 1
        self.audio_dir = os.getcwd()
        logger.info(f'Looking for audio files in: {self.audio_dir}')
        dec_files = []
        enc_files = []
        for filename in os.listdir(self.audio_dir):
            if filename.startswith('dump-') and filename.endswith('.wav'):
                filepath = os.path.join(self.audio_dir, filename)
                file_time = os.path.getmtime(filepath)
                if '-dec.wav' in filename:
                    dec_files.append((filepath, file_time))
                elif '-enc.wav' in filename:
                    enc_files.append((filepath, file_time))
        if dec_files:
            dec_files.sort(key=lambda x: x[1], reverse=True)
            self.current_dec_file = dec_files[0][0]
            logger.info(f'Selected decoded audio file: {self.current_dec_file}')
        if enc_files:
            enc_files.sort(key=lambda x: x[1], reverse=True)
            self.current_enc_file = enc_files[0][0]
            logger.info(f'Selected encoded audio file: {self.current_enc_file}')

    async def _create_tts_queue(self):
        """Create the TTS audio queue in the main event loop."""
        if self.tts_audio_queue is None:
            self.tts_audio_queue = asyncio.Queue()
            logger.debug('TTS audio queue created')

    async def _start_tts_sender(self):
        """Start the TTS audio sender task."""
        if self.tts_sender_task is None or self.tts_sender_task.done():
            self.tts_sender_task = asyncio.create_task(self._tts_audio_sender())
            logger.debug('TTS audio sender task started')

    async def _tts_audio_sender(self):
        """Background task to send TTS audio to the call."""
        logger.info('TTS audio sender started')
        try:
            while self.call_established:
                try:
                    if self.tts_audio_queue is None:
                        await asyncio.sleep(0.1)
                        continue
                    audio_chunk = await asyncio.wait_for(self.tts_audio_queue.get(), timeout=1.0)
                    if audio_chunk is None:
                        break
                    await self.send_tts_audio(audio_chunk)
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f'Error in TTS audio sender: {e}')
                    break
        except asyncio.CancelledError:
            logger.info('TTS audio sender cancelled')
        finally:
            logger.info('TTS audio sender stopped')

    async def send_tts_audio(self, audio_chunk: bytes):
        """Send TTS audio chunk to the call via JACK."""
        await self.audio_handler.send_tts_audio(audio_chunk)

    async def queue_tts_audio(self, audio_chunk: bytes):
        """Queue TTS audio chunk for sending to the call."""
        if self.call_established and self.tts_audio_queue is not None:
            try:
                await self.tts_audio_queue.put(audio_chunk)
            except Exception as e:
                logger.error(f'Failed to queue TTS audio: {e}')

    async def hangup_call(self):
        """Initiate call hangup and ensure cleanup is performed."""
        logger.info('Hangup requested. Performing cleanup...')
        self.handle_call_ended('Hangup command received')
        self.hang()

    def handle_call_ended(self, reason):
        """When call ends, cleanup resources."""
        logger.info('=== CALL ENDED ===')
        if self.audio_capture:
            self._schedule_coroutine(self.audio_capture.stop())
        if self.stt:
            self._schedule_coroutine(self.stt.stop())
        if self.tts_sender_task:
            self.tts_sender_task.cancel()
        import time
        time.sleep(1)
        if not self.current_enc_file and (not self.current_dec_file):
            logger.info('Audio files not set, searching for them now...')
            try:
                import time
                self.audio_dir = os.getcwd()
                logger.info(f'Looking for audio files in: {self.audio_dir}')
                dec_files = []
                enc_files = []
                for filename in os.listdir(self.audio_dir):
                    if filename.startswith('dump-') and filename.endswith('.wav'):
                        filepath = os.path.join(self.audio_dir, filename)
                        file_time = os.path.getmtime(filepath)
                        if '-dec.wav' in filename:
                            dec_files.append((filepath, file_time))
                        elif '-enc.wav' in filename:
                            enc_files.append((filepath, file_time))
                if dec_files:
                    dec_files.sort(key=lambda x: x[1], reverse=True)
                    self.current_dec_file = dec_files[0][0]
                    logger.info(f'Found decoded audio file: {self.current_dec_file}')
                if enc_files:
                    enc_files.sort(key=lambda x: x[1], reverse=True)
                    self.current_enc_file = enc_files[0][0]
                    logger.info(f'Found encoded audio file: {self.current_enc_file}')
            except Exception as e:
                logger.error(f'Error finding audio files: {e}')
                import traceback
                logger.error(traceback.format_exc())
        logger.info(f'Audio file status - enc: {self.current_enc_file}, dec: {self.current_dec_file}')
        input_file = None
        if self.current_dec_file and os.path.exists(self.current_dec_file):
            input_file = self.current_dec_file
            logger.info(f'Using dec file as input: {input_file}')
        elif self.current_enc_file and os.path.exists(self.current_enc_file):
            input_file = self.current_enc_file
            logger.info(f'Using enc file as input: {input_file}')
        if input_file:
            try:
                import time
                time.sleep(0.5)
                logger.info('Waited 0.5s for files to be fully written')
                output_path = self.audio_combiner.get_output_filename(log_id=self.context.log_id if self.context else None, base_dir='data/calls')
                logger.info(f'Target output path: {output_path}')
                logger.info(f'Attempting to combine audio from: {input_file}')
                success = self.audio_combiner.combine_from_single_file(input_file=input_file, output_path=output_path)
                if success:
                    logger.info(f'✓ Combined call recording saved to: {output_path}')
                    if os.path.exists(output_path):
                        size = os.path.getsize(output_path)
                        logger.info(f'✓ Combined file size: {size:,} bytes')
                    else:
                        logger.error(f'✗ Output file does not exist after combine: {output_path}')
                else:
                    logger.error(f'✗ Failed to combine audio files')
            except Exception as e:
                logger.error(f'Failed to combine call audio files: {e}')
        else:
            logger.warning('No audio files found to combine')
        self.audio_handler.cleanup(self)
        if self.current_dec_file and os.path.exists(self.current_dec_file):
            size = os.path.getsize(self.current_dec_file)
            logger.info(f'Incoming audio saved to: {self.current_dec_file}')
            logger.info(f'File size: {size:,} bytes')
        if self.utterances:
            logger.info(f'Call transcript: {len(self.utterances)} utterances captured')
            for utterance in self.utterances[-3:]:
                logger.info(f"[{utterance['time_str']}] {utterance['text']}")
        else:
            logger.info('No utterances detected.')
        if self.stt:
            stats = self.stt.get_stats()
            logger.info(f'STT Stats: {stats}')
        self._schedule_coroutine(self.show_disconnected())

    def get_transcript(self):
        """Get full transcript as a single string."""
        return '\n'.join([u['text'] for u in self.utterances])

    async def show_disconnected(self):
        msg = '\n\nSYSTEM:  -- CALL DISCONNECTED  --\n\n'
        await service_manager.backend_user_message(message=msg)
        await service_manager.send_message_to_agent(session_id=self.context.log_id, message=msg, context=self.context)

    def get_utterances(self):
        """Get all captured utterances."""
        return self.utterances

def setup_sndfile_module():
    """Helper function to set up the sndfile module in baresip config."""
    config_path = os.path.expanduser('~/.baresip/config')
    if not os.path.exists(config_path):
        logger.warning('Baresip config not found. Run baresip once to create it.')
        return False
    with open(config_path, 'r') as f:
        config = f.read()
    if 'module\t\t\tsndfile.so' in config:
        logger.info('sndfile module already enabled')
        return True
    elif '#module\t\t\tsndfile.so' in config:
        config = config.replace('#module\t\t\tsndfile.so', 'module\t\t\tsndfile.so')
        with open(config_path, 'w') as f:
            f.write(config)
        logger.info('Enabled sndfile module in config')
        return True
    elif 'module_path' in config:
        config = config.replace('module_path\t\t/usr/local/lib/baresip/modules', 'module_path\t\t/usr/local/lib/baresip/modules\nmodule\t\t\tsndfile.so')
        with open(config_path, 'w') as f:
            f.write(config)
        logger.info('Added sndfile module to config')
        return True
    else:
        logger.warning('Could not automatically enable sndfile module.')
        logger.warning(f"Please add 'module\t\t\tsndfile.so' to {config_path}")
        return False