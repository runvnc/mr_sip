"""
SIP Client for Deepgram STT Mode using PySIP (V2)

This replaces the baresip+JACK implementation with pure Python PySIP.
Audio flows directly between PySIP RTP and Deepgram STT.

Key features:
- Uses PySIP for SIP signaling and RTP (no baresip, no JACK)
- Sends ulaw 8kHz audio directly to Deepgram (mulaw encoding, no conversion needed)
- Receives TTS audio and converts to ulaw 8kHz for RTP output
- Supports call recording
"""
import asyncio
import logging
import queue
import traceback
import time
import audioop
import os
from datetime import datetime
from lib.pipelines.pipe import pipeline_manager
from lib.providers.hooks import hook_manager
from typing import Optional, Callable
from PySIP.sip_call import SipCall
from PySIP.filters import CallState
from lib.providers.services import service_manager
from .call_recorder import CallRecorder, S2SBufferedRecorder
from .simple_recorder import SimpleRecorder
from .stt import create_stt_provider, BaseSTTProvider, STTResult
logger = logging.getLogger(__name__)

# End-to-end latency log (shared across mr_sip + PySIP)
E2E_LATENCY_LOG = '/tmp/sip_e2e_latency.log'


def _e2e_log(event: str, utterance_num: int = 0, **kwargs):
    """Log an end-to-end latency event with perf_counter timestamp."""
    from datetime import datetime
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

class AudioStreamAdapter:
    """Adapter to feed audio to PySIP's RTP session.
    
    PySIP expects an object with an input_q attribute (queue.Queue)
    that it reads audio frames from.
    """

    def __init__(self):
        self.input_q = queue.Queue(maxsize=1000)
        self.stream_id = f'tts_output_{time.time_ns()}'
        self._done = False
        self.pre_encoded = True

    def stream_done(self):
        """Mark stream as done."""
        self._done = True

class MindRootSIPBotV2:
    """SIP phone bot for Deepgram STT mode using PySIP.
    
    Handles bidirectional audio:
    - Input: Phone audio (ulaw 8kHz) -> Deepgram STT -> utterance callback
    - Output: TTS audio -> convert to ulaw 8kHz -> Phone
    """

    def __init__(self, user: str, password: str, gateway: str, audio_dir: str='.', on_utterance_callback: Callable=None, stt_provider: str=None, stt_config: dict=None, context=None, enable_recording: bool=False, recording_dir: str='recordings', record_separate: bool=False):
        """
        Args:
            user: SIP username
            password: SIP password
            gateway: SIP gateway (format: "host:port")
            audio_dir: Unused, kept for compatibility
            on_utterance_callback: Async function called with each complete utterance
                                  Signature: async callback(text, utterance_num, timestamp, context, is_eager)
            stt_provider: STT provider name ('deepgram_flux', 'deepgram', etc.)
            stt_config: Additional configuration for STT provider
            context: MindRoot ChatContext
            enable_recording: Enable call recording
            recording_dir: Directory to save recordings
            record_separate: If True, save separate incoming/outgoing files
        """
        self.sip_username = user
        self.sip_password = password
        if ':' in gateway:
            self.sip_server = gateway
        else:
            self.sip_server = f'{gateway}:5060'
            logger.info(f'No port specified in gateway, using default: {self.sip_server}')
        self.context = context
        self.on_utterance_callback = on_utterance_callback
        self.stt_provider_name = stt_provider or os.getenv('STT_PROVIDER', 'deepgram_flux')
        self.stt_config = stt_config or {}
        self.stt: Optional[BaseSTTProvider] = None
        self.call: Optional[SipCall] = None
        self.is_active = False
        self.call_established = False
        self.call_start_time: Optional[datetime] = None
        self.audio_stream: Optional[AudioStreamAdapter] = None
        self._input_frame_count = 0
        self._output_frame_count = 0
        self._dropped_frame_count = 0
        self._frame_remainder = b""  # leftover bytes waiting to complete a 160-byte frame
        self._tts_response_active = False
        self._tts_response_seq = 0
        self.call_answered = asyncio.Event()
        self.enable_recording = enable_recording
        self.recording_dir = recording_dir
        self.record_separate = record_separate
        self.recorder: Optional[CallRecorder] = None
        self._interrupting = False
        self.utterances = []
        self.last_partial_text = ''
        self.last_eager_eot_text = ''
        self.draft_response_active = False
        try:
            self.main_loop = asyncio.get_running_loop()
        except RuntimeError:
            self.main_loop = None
        finally:
            pass
        self.last_activity_time = time.time()
        self.silence_threshold = 200
        self.silence_reported = False
        self._s2s_active = True
        self._silence_monitor_task = None
        self._aborted = False
        logger.info(f'PySIP V2 Bot initialized for user {user} on gateway {gateway}')
        logger.info(f'STT provider: {self.stt_provider_name}')

    def wait_until_ready(self):
        """Compatibility method - PySIP doesn't need explicit ready wait."""
        pass

    def _schedule_coroutine(self, coro):
        """Schedule a coroutine to run in the main event loop from any thread."""
        if self.main_loop and (not self.main_loop.is_closed()):
            try:
                future = asyncio.run_coroutine_threadsafe(coro, self.main_loop)
                return future
            except Exception as e:
                logger.error(f'Failed to schedule coroutine: {e}')
            finally:
                pass
        else:
            pass
        return None

    async def _setup_stt(self):
        """Initialize and start the STT provider."""
        try:
            logger.info(f'Creating STT provider: {self.stt_provider_name}')
            stt_config = self.stt_config.copy()
            if self.stt_provider_name in ['deepgram', 'deepgram_flux']:
                stt_config['encoding'] = 'mulaw'
                stt_config['sample_rate'] = 8000
            else:
                pass
            self.stt = create_stt_provider(self.stt_provider_name, **stt_config)
            self.stt.set_callbacks(on_partial=self._on_partial_result, on_final=self._on_final_result)
            if hasattr(self.stt, 'set_turn_resumed_callback'):
                self.stt.set_turn_resumed_callback(self._handle_turn_resumed)
                logger.info('Barge-in detection enabled')
            else:
                pass
            if hasattr(self.stt, 'main_loop'):
                self.stt.main_loop = self.main_loop
            else:
                pass
            if hasattr(self.stt, 'set_sip_call_established'):
                self.stt.set_sip_call_established(True)
            else:
                pass
            await self.stt.start()
            logger.info(f'STT provider started: {self.stt_provider_name}')
        except Exception as e:
            logger.error(f'Error setting up STT: {e}')
            logger.error(traceback.format_exc())
            raise
        finally:
            pass

    @staticmethod
    def _stt_dlog(msg: str):
        """Write to the STT debug log so eager/final handling is visible in one place."""
        now = datetime.now()
        ts = now.strftime('%Y-%m-%d %H:%M:%S') + f'.{now.microsecond // 1000:03d}'
        line = f'[{ts}] [SIP] {msg}'
        try:
            with open('/tmp/silero_cohere_stt.log', 'a') as f:
                f.write(line + '\n')
                f.flush()
        except Exception:
            pass

    def _on_partial_result(self, result: STTResult):
        """Callback for partial transcription results."""
        if result.text != self.last_partial_text:
            logger.info(f'[PARTIAL] {result.text} (confidence: {result.confidence:.2f}, eager_eot: {result.is_eager_eot})')
            self.last_partial_text = result.text
            if hasattr(result, 'is_eager_eot') and result.is_eager_eot:
                logger.info(f'[EAGER EOT] Starting AI response preparation for: {result.text}')
                self.draft_response_active = True
                self.last_eager_eot_text = result.text
                self._stt_dlog(f'[EAGER EOT] Calling utterance callback (is_eager=True): "{result.text}"')
                _e2e_log('EAGER_EOT_CALLBACK', utterance_num=len(self.utterances) + 1, text=result.text[:50])
                if self.on_utterance_callback:
                    utterance_num = len(self.utterances) + 1
                    self._schedule_coroutine(self._call_utterance_callback(result.text, utterance_num, result.timestamp or time.time(), is_eager=True))
                    self._stt_dlog(f'[EAGER EOT] Callback scheduled for: "{result.text}"')
                else:
                    pass
            else:
                pass
        else:
            pass

    def _on_final_result(self, result: STTResult):
        """Callback for final transcription results."""
        utterance_data = {'number': result.utterance_num or len(self.utterances) + 1, 'text': result.text, 'timestamp': result.timestamp or time.time(), 'confidence': result.confidence, 'time_str': time.strftime('%H:%M:%S', time.localtime(result.timestamp or time.time()))}
        self.utterances.append(utterance_data)
        logger.info(f"[{utterance_data['time_str']}] Utterance #{utterance_data['number']}: {result.text}")
        if result.text == self.last_eager_eot_text and self.last_eager_eot_text:
            logger.info(f"[FINAL] Skipping duplicate - already sent via EagerEOT: '{result.text}'")
            self._stt_dlog(f'[FINAL] SKIPPED (already sent via eager): "{result.text}"')
        else:
            logger.info(f"[FINAL] Sending to agent: '{result.text}'")
            self._stt_dlog(f'[FINAL] Sending to agent (no eager match): "{result.text}"')
            if self.on_utterance_callback:
                self._schedule_coroutine(self._call_utterance_callback(result.text, utterance_data['number'], utterance_data['timestamp'], is_eager=False))
            else:
                pass
        self.draft_response_active = False
        self.last_eager_eot_text = ''
        self.last_partial_text = ''
        self._schedule_coroutine(hook_manager.user_utterance_complete(text=result.text, context=self.context))

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
        finally:
            pass

    def _handle_turn_resumed(self):
        """Handle TurnResumed event from Deepgram - user is speaking (barge-in)."""
        logger.info('[BARGE-IN] User speaking - clearing audio queue')
        self._schedule_coroutine(self._halt_audio_output())
        self.last_eager_eot_text = ''
        if self.draft_response_active:
            logger.info('[TURN RESUMED] Cancelling draft AI response')
            self._schedule_coroutine(self._cancel_ai_response())
        else:
            self.clear_audio_queue()
        self.draft_response_active = False

    async def _cancel_ai_response(self):
        """Cancel active AI response using service manager."""
        if not self.context or not self.context.log_id:
            logger.warning('Cannot cancel AI response: no context or log_id')
            return
        else:
            pass
        try:
            result = await service_manager.cancel_active_response(log_id=self.context.log_id, context=self.context)
            logger.info(f'AI response cancelled: {result}')
        except Exception as e:
            logger.error(f'Error cancelling AI response: {e}')
        finally:
            pass

    async def _halt_audio_output(self):
        """Halt audio output using service manager."""
        if not self.context or not self.context.log_id:
            logger.warning('Cannot halt audio: no context or log_id')
            self.clear_audio_queue()
            return
        else:
            pass
        try:
            result = await service_manager.sip_halt_audio(context=self.context)
            logger.info(f'Audio halt result: {result}')
        except Exception as e:
            logger.error(f'Error halting audio: {e}')
            self.clear_audio_queue()
        finally:
            pass

    async def make_call(self, destination: str):
        """Initiate outbound call.
        
        Args:
            destination: Phone number or SIP URI to call
        """
        logger.info(f'=== INITIATING CALL TO {destination} (PySIP V2 + Deepgram) ===')
        try:
            self.call = SipCall(username=self.sip_username, password=self.sip_password, route=self.sip_server, callee=destination)
            logger.info(f'SipCall instance created for {destination}')

            @self.call.on_call_state_changed
            async def on_state(state: CallState):
                try:
                    logger.info(f'Call state changed: {state}')
                    if state in [CallState.ENDED, CallState.FAILED, CallState.BUSY]:
                        await self._on_call_ended(state)
                    else:
                        pass
                except Exception as e:
                    logger.error(f'Error in on_call_state_changed callback: {e}')
                    logger.error(traceback.format_exc())
                finally:
                    pass

            @self.call.on_frame_received
            async def on_frame(frame):
                """Receive audio from phone and send to Deepgram STT.
                
                PySIP provides ulaw 8kHz frames (typically 160 bytes = 20ms).
                We send these directly to Deepgram with mulaw encoding.
                """
                try:
                    rtp_ts = None
                    if self._aborted:
                        logger.warning('on_frame called but bot is aborted (timeout cleanup) - ignoring frame and hanging up')
                        if self.call:
                            await self.call.stop('Aborted - call connected after timeout')
                        return

                    if hasattr(frame, 'data'):
                        ulaw_bytes = frame.data
                        rtp_ts = getattr(frame, 'timestamp', None)
                    else:
                        ulaw_bytes = frame
                    if not self.call_established and self.call and self.call._rtp_session:
                        # Do not attach an outbound AudioStreamAdapter here.
                        # Outbound streams are now created per TTS response by
                        # start_tts_response(), so PySIP prebuffers at speak()
                        # boundaries instead of only once at call readiness.
                        self.is_active = True
                        self.call_established = True
                        self.call_start_time = datetime.now()
                        await self._setup_stt()
                        self.last_activity_time = time.time()
                        self._silence_monitor_task = asyncio.create_task(self._monitor_silence())
                        self.call_answered.set()
                        logger.info('Call fully answered and ready for audio')
                        if self.enable_recording:
                            self.recorder = SimpleRecorder(self.context.log_id, self.recording_dir, record_separate=self.record_separate, record_combined=True)
                            await self.recorder.start_recording()
                        else:
                            pass
                    else:
                        pass
                    self._input_frame_count += 1
                    try:
                        pcm_data = audioop.ulaw2lin(ulaw_bytes, 2)
                        rms = audioop.rms(pcm_data, 2)
                        if rms > self.silence_threshold:
                            self.last_activity_time = time.time()
                            if self.silence_reported:
                                self.silence_reported = False
                            else:
                                pass
                        else:
                            pass
                    except Exception as e:
                        self.last_activity_time = time.time()
                    finally:
                        pass
                    if self._input_frame_count % 50 == 0:
                        logger.debug(f'Received frame #{self._input_frame_count}, size: {len(ulaw_bytes)} bytes')
                    else:
                        pass
                    if self.recorder:
                        if rtp_ts is not None:
                            self.recorder.record_incoming_with_timestamp(ulaw_bytes, rtp_ts)
                        else:
                            self.recorder.record_incoming(ulaw_bytes)
                    else:
                        pass
                    # Only run sip_audio_in pipeline if explicitly enabled (used by mr_csm_stream)
                    # TODO: Replace with subscribe_sip_audio_in/unsubscribe_sip_audio_in service
                    # that lets plugins register an asyncio.Queue directly on PySIP's RTP
                    # _output_queues for zero per-frame overhead. See mr_csm_stream/mod.py
                    # for details.
                    if os.environ.get('MR_SIP_AUDIO_IN_PIPELINE', '').lower() in ('1', 'true', 'yes'):
                        try:
                            await pipeline_manager.execute_pipeline('sip_audio_in', {'audio_bytes': ulaw_bytes, 'timestamp': time.time()}, context=self.context)
                        except Exception as e:
                            pass
                        finally:
                            pass
                    if self.stt and self.stt.is_running:
                        if hasattr(self.stt, 'add_audio_bytes'):
                            await self.stt.add_audio_bytes(ulaw_bytes)
                        else:
                            pcm_data = audioop.ulaw2lin(ulaw_bytes, 2)
                            audio_array = np.frombuffer(pcm_data, dtype=np.int16)
                            audio_float = audio_array.astype(np.float32) / 32768.0
                            await self.stt.add_audio(audio_float)
                    else:
                        pass
                except Exception as e:
                    logger.error(f'Error in on_frame_received callback: {e}')
                    logger.error(traceback.format_exc())
                finally:
                    pass
            logger.info('Callbacks registered, starting PySIP call...')
            await self.call.start()
            logger.info('PySIP call.start() completed')
        except Exception as e:
            logger.error(f'Error in make_call: {e}')
            logger.error(traceback.format_exc())
            raise
    async def attach_to_incoming_call(self, call):
        """Attach this bot to an already-accepted incoming SipCall.
        
        This is used for incoming calls where PySIP has already handled
        the INVITE/100/180/200 handshake. We just need to wire up the
        audio callbacks and STT.
        """
        logger.info(f'=== ATTACHING TO INCOMING CALL (PySIP V2) ===')
        self.call = call
        
        # Mark call as active and set up STT immediately so TTS and
        # transcription work from the very start of the call.
        self.is_active = True
        self.call_established = True
        self.call_start_time = datetime.now()
        self.last_activity_time = time.time()
        await self._setup_stt()
        self._silence_monitor_task = asyncio.create_task(self._monitor_silence())
        self.call_answered.set()
        logger.info('Incoming call marked active, STT started')
        if self.enable_recording:
            self.recorder = SimpleRecorder(self.context.log_id, self.recording_dir, record_separate=self.record_separate, record_combined=True)
            await self.recorder.start_recording()
        
        @call.on_call_state_changed
        async def on_state(state):
            try:
                logger.info(f'Incoming call state changed: {state}')
                if state in [CallState.ENDED, CallState.FAILED, CallState.BUSY]:
                    await self._on_call_ended(state)
            except Exception as e:
                logger.error(f'Error in incoming call state callback: {e}')
                logger.error(traceback.format_exc())
        
        @call.on_frame_received
        async def on_frame(frame):
            try:
                rtp_ts = None
                if hasattr(frame, 'data'):
                    ulaw_bytes = frame.data
                    rtp_ts = getattr(frame, 'timestamp', None)
                else:
                    ulaw_bytes = frame
                
                # Call is already active from attach_to_incoming_call setup
                
                self._input_frame_count += 1
                try:
                    pcm_data = audioop.ulaw2lin(ulaw_bytes, 2)
                    rms = audioop.rms(pcm_data, 2)
                    if rms > self.silence_threshold:
                        self.last_activity_time = time.time()
                        if self.silence_reported:
                            self.silence_reported = False
                except Exception:
                    self.last_activity_time = time.time()
                
                if self._input_frame_count % 50 == 0:
                    logger.debug(f'Received frame #{self._input_frame_count}, size: {len(ulaw_bytes)} bytes')
                
                if self.recorder:
                    if rtp_ts is not None:
                        self.recorder.record_incoming_with_timestamp(ulaw_bytes, rtp_ts)
                    else:
                        self.recorder.record_incoming(ulaw_bytes)
                
                if os.environ.get('MR_SIP_AUDIO_IN_PIPELINE', '').lower() in ('1', 'true', 'yes'):
                    try:
                        await pipeline_manager.execute_pipeline('sip_audio_in', {'audio_bytes': ulaw_bytes, 'timestamp': time.time()}, context=self.context)
                    except Exception:
                        pass
                
                if self.stt and self.stt.is_running:
                    if self._input_frame_count <= 3:
                        logger.warning(f'INCOMING STT: Sending frame #{self._input_frame_count}, {len(ulaw_bytes)} bytes, stt_running={self.stt.is_running}, stt_conn={getattr(self.stt, connection, None) is not None}')
                    if hasattr(self.stt, 'add_audio_bytes'):
                        await self.stt.add_audio_bytes(ulaw_bytes)
                    else:
                        pcm_data = audioop.ulaw2lin(ulaw_bytes, 2)
                        audio_array = np.frombuffer(pcm_data, dtype=np.int16)
                        audio_float = audio_array.astype(np.float32) / 32768.0
                        await self.stt.add_audio(audio_float)
                elif self._input_frame_count <= 3:
                    logger.warning(f'INCOMING STT: Frame #{self._input_frame_count} but stt={self.stt}, is_running={getattr(self.stt, is_running, None)}')
            except Exception as e:
                logger.error(f'Error in incoming call frame callback: {e}')
                logger.error(traceback.format_exc())
        
        logger.info('Incoming call callbacks registered')

    async def _on_call_ended(self, state: CallState):
        """Called when call ends."""
        try:
            logger.info(f'=== CALL ENDED: {state} (PySIP V2) ===')
            self.is_active = False
            self.call_established = False
            if self._silence_monitor_task:
                self._silence_monitor_task.cancel()
                try:
                    await self._silence_monitor_task
                except asyncio.CancelledError:
                    pass
                finally:
                    pass
                self._silence_monitor_task = None
            else:
                pass
            if self.stt:
                await self.stt.stop()
                self.stt = None
            else:
                pass
            if self.audio_stream:
                try:
                    self.audio_stream.input_q.put(None, block=False)
                    self.audio_stream.stream_done()
                except Exception as e:
                    logger.warning(f'Error stopping audio stream: {e}')
                finally:
                    pass
            else:
                pass
            if self.recorder:
                try:
                    self.recorder.interrupt_outgoing()
                    self.recorder.interrupt_incoming()
                except Exception:
                    pass
                finally:
                    pass
                await self.recorder.stop_recording()
                self.recorder = None
            else:
                pass
            await self._show_disconnected()
            logger.info(f'Call statistics - Input frames: {self._input_frame_count}, Output frames: {self._output_frame_count}, Dropped frames: {self._dropped_frame_count}')
            if self.stt:
                stats = self.stt.get_stats()
                logger.info(f'STT Stats: {stats}')
            else:
                pass
        except Exception as e:
            logger.error(f'Error in _on_call_ended: {e}')
            logger.error(traceback.format_exc())
        finally:
            pass

    async def start_tts_response(self) -> bool:
        """Start a fresh outbound TTS response stream for PySIP.

        PySIP's outgoing prebuffer is keyed by audio stream identity.  A new
        AudioStreamAdapter per response lets PySIP prebuffer at the beginning
        of each AI response instead of only once at call setup.
        """
        try:
            if not self.is_active:
                logger.warning('Cannot start TTS response - call not active')
                return False
            if not self.call or not getattr(self.call, '_rtp_session', None):
                logger.warning('Cannot start TTS response - RTP session not initialized')
                return False

            self._tts_response_seq += 1
            stream = AudioStreamAdapter()
            stream.stream_id = f'tts_output_{self._tts_response_seq}_{time.time_ns()}'

            # Drop any partial frame from a previous response so every response
            # begins on a clean 20ms/160-byte ulaw boundary.
            self._frame_remainder = b""

            self.audio_stream = stream
            self.call._rtp_session.set_audio_stream(stream)
            self._tts_response_active = True
            self._tts_response_start_pc = time.perf_counter()
            # Reset per-response frame counter for FIRST_TTS_CHUNK_PYSIP logging
            self._response_output_frame_count = 0
            # Store current utterance num for downstream e2e logging
            self._e2e_current_utterance_num = getattr(self.stt, '_utterance_count', 0) if self.stt else 0
            # Attach VAD eager end timestamp so PySIP can compute e2e latency
            # when the first RTP packet is sent
            if self.stt and hasattr(self.stt, '_last_vad_eager_end_pc'):
                stream._e2e_vad_eager_end_pc = self.stt._last_vad_eager_end_pc
                # True user-perceived start: last speech audio time (before VAD silence gap)
                if hasattr(self.stt, '_last_user_speech_end_pc'):
                    stream._e2e_user_speech_end_pc = self.stt._last_user_speech_end_pc
                else:
                    stream._e2e_user_speech_end_pc = self.stt._last_vad_eager_end_pc
                stream._e2e_vad_utterance_num = self._e2e_current_utterance_num
            else:
                stream._e2e_vad_eager_end_pc = None
                stream._e2e_user_speech_end_pc = None
                stream._e2e_vad_utterance_num = 0
            _e2e_log('TTS_RESPONSE_START', utterance_num=self._e2e_current_utterance_num)
            logger.debug(f'Started TTS response stream {stream.stream_id}')
            return True
        except Exception as e:
            logger.error(f'Error starting TTS response stream: {e}')
            logger.error(traceback.format_exc())
            return False

    async def end_tts_response(self) -> bool:
        """Finish the current outbound TTS response stream.

        We enqueue a None sentinel and let PySIP drain queued frames.  Do not
        detach the stream immediately here or the RTP sender may drop the tail
        of the response before it has drained its queue/smoothing buffer.
        """
        try:
            stream = self.audio_stream
            if not stream:
                self._tts_response_active = False
                self._frame_remainder = b""
                return False

            if self._frame_remainder:
                # Pad final partial 20ms frame with PCMU silence.  All current
                # SIP TTS providers feed pre-encoded ulaw_8000 audio.
                final_frame = self._frame_remainder.ljust(160, b"\xff")[:160]
                self._frame_remainder = b""
                try:
                    stream.input_q.put(final_frame, block=True, timeout=0.5)
                except queue.Full:
                    logger.warning('TTS response stream full while flushing final partial frame')

            try:
                stream.input_q.put(None, block=False)
            except queue.Full:
                logger.warning('TTS response stream full while enqueueing end sentinel')
            stream.stream_done()
            self._tts_response_active = False
            logger.debug(f'Ended TTS response stream {stream.stream_id}')
            return True
        except Exception as e:
            logger.error(f'Error ending TTS response stream: {e}')
            logger.error(traceback.format_exc())
            self._tts_response_active = False
            return False

    async def send_tts_audio(self, audio_chunk: bytes, timestamp=None):
        """Send TTS audio chunk to the SIP call.
        
        Handles format detection and conversion to ulaw 8kHz.
        
        Args:
            audio_chunk: Audio data (PCM or ulaw, various sample rates)
            timestamp: Optional timestamp for audio pacing
        """
        try:
            if not self.is_active:
                logger.warning('Cannot send audio - call not active')
                return
            else:
                pass
            if not self._tts_response_active:
                # Backward-compatible fallback for callers that only know about
                # sip_audio_out_chunk (not explicit response lifecycle).
                await self.start_tts_response()
            else:
                pass
            if not self.audio_stream:
                logger.warning('Cannot send audio - audio stream not initialized')
                return
            else:
                pass
            chunk_duration = len(audio_chunk) / 8000.0
            self.last_activity_time = max(self.last_activity_time, time.time()) + chunk_duration
            if self.silence_reported:
                self.silence_reported = False
            else:
                pass
            ulaw_audio = self._convert_to_ulaw(audio_chunk)

            # Prepend leftover bytes from previous chunk so frames are always
            # exactly 160 bytes. Short frames cause clicks/pops at the receiver.
            ulaw_audio = self._frame_remainder + ulaw_audio
            n_complete = (len(ulaw_audio) // 160) * 160
            self._frame_remainder = ulaw_audio[n_complete:]
            ulaw_audio = ulaw_audio[:n_complete]

            if self._interrupting:
                return
            else:
                pass
            FRAME_SIZE = 160
            for i in range(0, len(ulaw_audio), FRAME_SIZE):
                if self._interrupting:
                    return
                if not hasattr(self, '_response_output_frame_count'):
                    self._response_output_frame_count = 0
                if self._response_output_frame_count == 0:
                    _e2e_log('FIRST_TTS_CHUNK_PYSIP', utterance_num=getattr(self, '_e2e_current_utterance_num', 0),
                             since_tts_response_start_ms=f'{(time.perf_counter() - getattr(self, "_tts_response_start_pc", time.perf_counter()))*1000:.0f}',
                             chunk_len=len(ulaw_audio[i:i + FRAME_SIZE]))
                self._response_output_frame_count += 1
                frame = ulaw_audio[i:i + FRAME_SIZE]
                frame_timestamp = timestamp + i / 8000.0 if timestamp else None
                try:
                    if frame_timestamp:
                        self.audio_stream.input_q.put((frame, frame_timestamp), block=True, timeout=0.5)
                    else:
                        self.audio_stream.input_q.put(frame, block=True, timeout=0.5)
                    if self.recorder:
                        try:
                            self.recorder.record_outgoing(frame, timestamp=frame_timestamp)
                        except TypeError:
                            self.recorder.record_outgoing(frame)
                        finally:
                            pass
                    else:
                        pass
                    self._output_frame_count += 1
                except queue.Full:
                    logger.critical('Audio queue full!')
                    self._dropped_frame_count += 1
                finally:
                    pass
            else:
                pass
        except Exception as e:
            logger.error(f'Error in send_tts_audio: {e}')
            logger.error(traceback.format_exc())
        finally:
            pass

    def _convert_to_ulaw(self, audio_chunk: bytes) -> bytes:
        """Pass through audio - assumes input is already ulaw 8kHz.
        
        ElevenLabs is configured to output ulaw_8000 format directly,
        so no conversion is needed. This method exists for compatibility
        and potential future use with other TTS providers that may send
        different formats.
        
        Args:
            audio_chunk: Audio data (expected to be ulaw 8kHz from ElevenLabs)
        
        Returns:
            The audio chunk unchanged
        """
        return audio_chunk

    def clear_audio_queue(self):
        """Clear all queued audio frames (for interruption)."""
        try:
            self._tts_response_active = False
            self._frame_remainder = b""
            self._interrupting = True
            self.last_activity_time = time.time()
            if self.recorder:
                try:
                    self.recorder.interrupt_outgoing()
                except Exception:
                    pass
                finally:
                    pass
            else:
                pass
            if self.audio_stream:
                try:
                    cleared_count = 0
                    while not self.audio_stream.input_q.empty():
                        try:
                            self.audio_stream.input_q.get_nowait()
                            cleared_count += 1
                        except:
                            break
                        finally:
                            pass
                    else:
                        pass
                    logger.info(f'Cleared {cleared_count} audio frames from queue')
                except Exception as e:
                    logger.error(f'Error clearing audio queue: {e}')
                finally:
                    pass
            else:
                pass
            if self.call and hasattr(self.call, '_rtp_session') and self.call._rtp_session:
                self.call._rtp_session.__outgoing_buffer = []
            else:
                pass
        finally:
            self._interrupting = False

    async def hangup_call(self):
        """Initiate call hangup and cleanup."""
        try:
            logger.info('Hangup requested. Performing cleanup...')
            if self.call:
                await self.call.stop('Agent hangup')
                logger.info('Call stop requested')
            else:
                logger.warning('No active call to hang up')
        except Exception as e:
            logger.error(f'Error in hangup_call: {e}')
            logger.error(traceback.format_exc())
        finally:
            pass

    async def _monitor_silence(self):
        """Monitor for silence on both channels."""
        try:
            while self.is_active:
                await asyncio.sleep(0.5)
                duration = time.time() - self.last_activity_time
                if duration > 10.0 and (not self.silence_reported):
                    self.silence_reported = True
                    msg = f'[SYSTEM: No audio detected for {duration:.1f} seconds.]'
                    logger.info(f'Silence detected: {msg}')
                    try:
                        await service_manager.backend_user_message(message=msg, context=self.context)
                        await service_manager.send_message_to_agent(session_id=self.context.log_id, message=msg, context=self.context)
                    except Exception as e:
                        logger.warning(f'Failed to send silence notification: {e}')
                        self._s2s_active = False
                        break
                    finally:
                        pass
                else:
                    pass
            else:
                pass
        except asyncio.CancelledError:
            logger.info('Silence monitor cancelled')
        except Exception as e:
            logger.error(f'Error in silence monitor: {e}')
        finally:
            pass

    async def _show_disconnected(self):
        """Send disconnect message to agent."""
        try:
            msg = '\n\nSYSTEM: -- CALL DISCONNECTED --\n\n'
            await service_manager.backend_user_message(message=msg, context=self.context)
            await service_manager.send_message_to_agent(session_id=self.context.log_id, message=msg, context=self.context)
            logger.info('Disconnect message sent to agent')
        except Exception as e:
            logger.error(f'Error sending disconnect message: {e}')
        finally:
            pass

    def hang(self):
        """Synchronous hangup method for compatibility."""
        try:
            if self.call:
                try:
                    loop = asyncio.get_event_loop()
                    if not loop.is_closed():
                        asyncio.create_task(self.hangup_call())
                    else:
                        pass
                except Exception as e:
                    logger.error(f'Error scheduling hangup: {e}')
                finally:
                    pass
            else:
                logger.warning('hang() called but no active call')
        except Exception as e:
            logger.error(f'Error in hang(): {e}')
        finally:
            pass

    def get_transcript(self):
        """Get full transcript as a single string."""
        return '\n'.join([u['text'] for u in self.utterances])

    def get_utterances(self):
        """Get all captured utterances."""
        return self.utterances

    def get_metrics(self) -> dict:
        """Get audio metrics for monitoring."""
        total_frames = max(1, self._output_frame_count)
        drop_rate = self._dropped_frame_count / total_frames * 100
        return {'input_frames': self._input_frame_count, 'output_frames': self._output_frame_count, 'dropped_frames': self._dropped_frame_count, 'drop_rate_percent': drop_rate, 'is_active': self.is_active, 'call_established': self.call_established}

def setup_sndfile_module():
    """Compatibility stub - not needed for PySIP."""
    return True
