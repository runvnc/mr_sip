"""
MindRoot SIP Plugin - Internal Services (V2 with PySIP + Deepgram STT)

This version uses PySIP for SIP/RTP handling instead of baresip+JACK.
Supports Deepgram Flux and other STT providers.
"""
import os
import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, Any
from lib.providers.services import service, service_manager
from lib.providers.hooks import hook
from .sip_manager import get_session_manager
from PySIP.filters import CallState
from .sip_client_v2 import MindRootSIPBotV2, setup_sndfile_module
from .sip_account_wrapper import MindRootSIPAccount
from dotenv import load_dotenv
load_dotenv()
logger = logging.getLogger(__name__)

# End-to-end latency log (shared across mr_sip + PySIP)
E2E_LATENCY_LOG = '/tmp/sip_e2e_latency.log'


def _e2e_log(event: str, utterance_num: int = 0, **kwargs):
    """Log an end-to-end latency event with perf_counter timestamp."""
    from datetime import datetime as _dt
    now = _dt.now()
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

def _is_emergency_number(number: str) -> bool:
    """
    Check if a number matches emergency patterns like 911.
    
    Args:
        number: The phone number to check
        
    Returns:
        bool: True if number matches emergency pattern
    """
    normalized = number.replace('-', '').replace('.', '').replace(' ', '')
    normalized = normalized.replace('(', '').replace(')', '')
    normalized = normalized.replace('+', '')
    emergency_patterns = ['911', '112', '999', '000', '119', '110', '118']
    for pattern in emergency_patterns:
        if normalized == pattern:
            return True
        else:
            pass
    else:
        pass
    if normalized.startswith('1') and len(normalized) > 1:
        normalized_without_country = normalized[1:]
        for pattern in emergency_patterns:
            if normalized_without_country == pattern:
                return True
            else:
                pass
        else:
            pass
    else:
        pass
    return False

@service()
async def dial_service_v2(destination: str, context=None) -> Dict[str, Any]:
    """
    Service to initiate SIP calls using PySIP with Deepgram STT.
    
    This V2 version uses PySIP for SIP/RTP (replacing baresip+JACK)
    and supports the abstract STT provider interface.
    
    Args:
        destination: Phone number or SIP URI to call
        context: MindRoot context (required for session linking)
    
    Returns:
        dict: Session information including log_id, destination, and status

    Environment Variables:
        SIP_GATEWAY: SIP gateway server (format: "host:port")
        SIP_USER: SIP username
        SIP_PASSWORD: SIP password
        STT_PROVIDER: 'deepgram_flux', 'deepgram', or 'whisper_vad' (default: 'deepgram_flux')
        DEEPGRAM_API_KEY: Required if using Deepgram
        SIP_ENABLE_RECORDING: Enable call recording (default: false)
        SIP_RECORDING_DIR: Directory for recordings (default: data/calls)
        SIP_RECORD_SEPARATE: Save separate incoming/outgoing files (default: false)
    """
    if not context or not context.log_id:
        raise ValueError('Context with log_id is required for SIP calls')
    else:
        pass
    if _is_emergency_number(destination):
        logger.warning(f'Emergency number dialing blocked: {destination} for session {context.log_id}')
        return {'status': 'blocked', 'log_id': context.log_id, 'destination': destination, 'error': 'Emergency number dialing is not permitted'}
    else:
        pass
    sip_gateway = os.getenv('SIP_GATEWAY', 'no sip gateway')
    sip_user = os.getenv('SIP_USER', 'nouser')
    sip_password = os.getenv('SIP_PASSWORD', 'no sip password')
    stt_provider = os.getenv('STT_PROVIDER', 'deepgram_flux')
    deepgram_api_key = os.getenv('DEEPGRAM_API_KEY', '')
    audio_dir = os.getenv('AUDIO_DIR', os.path.expanduser('.'))
    require_deepgram = os.getenv('REQUIRE_DEEPGRAM', 'true').lower() in ('true', '1', 'yes', 'on')
    is_local_provider = stt_provider in ('silero_cohere', 'smart_turn_v3', 'umut')
    call_establish_timeout = int(os.getenv('SIP_CALL_ESTABLISH_TIMEOUT', '120'))
    enable_recording = os.getenv('SIP_ENABLE_RECORDING', 'false').lower() == 'true'
    recording_dir = os.getenv('SIP_RECORDING_DIR', 'data/calls')
    record_separate = os.getenv('SIP_RECORD_SEPARATE', 'false').lower() == 'true'
    logger.info(f'Initiating PySIP call to {destination} for session {context.log_id}')
    logger.info(f'Using STT provider: {stt_provider}')
    if require_deepgram and not is_local_provider:
        if stt_provider not in ['deepgram', 'deepgram_flux']:
            error_msg = f"\n\n{'=' * 80}\nFATAL ERROR: Deepgram is required but STT_PROVIDER='{stt_provider}'\n{'=' * 80}\nPlease set: export STT_PROVIDER=deepgram_flux (recommended) or deepgram\nOr disable requirement: export REQUIRE_DEEPGRAM=false\n{'=' * 80}\n"
            logger.error(error_msg)
            import sys
            sys.exit(1)
        else:
            pass
        if not deepgram_api_key and not is_local_provider:
            error_msg = f"\n\n{'=' * 80}\nFATAL ERROR: DEEPGRAM_API_KEY environment variable not set\n{'=' * 80}\nDeepgram is required but no API key was provided.\n\nTo fix this:\n1. Get an API key from https://deepgram.com/\n2. Set it: export DEEPGRAM_API_KEY='your_key_here'\n\nOr to disable this requirement:\n   export REQUIRE_DEEPGRAM=false\n{'=' * 80}\n"
            logger.error(error_msg)
            import sys
            sys.exit(1)
        else:
            pass
    else:
        pass
    try:
        destination = ''.join((c for c in destination if c.isalnum() or c == '@'))
        if destination.isdigit() and len(destination) == 10:
            destination = '1' + destination
        else:
            pass
        if stt_provider in ['deepgram', 'deepgram_flux']:
            logger.info('\n' + '=' * 80)
            logger.info(f'INITIALIZING {stt_provider.upper()} STT PROVIDER')
            logger.info('=' * 80)
            logger.info(f"API Key: {deepgram_api_key[:10]}...{(deepgram_api_key[-4:] if len(deepgram_api_key) > 14 else '[too short]')}")
            logger.info(f'Destination: {destination}')
            logger.info(f'Session: {context.log_id}')
            logger.info('=' * 80)
        else:
            pass

        async def on_utterance_callback(text: str, utterance_num: int, timestamp: float, ctx, is_eager: bool=False):
            """Callback for when complete utterances are transcribed"""
            try:
                _e2e_log('UTTERANCE_CALLBACK', utterance_num=utterance_num, session=getattr(ctx, 'log_id', None) or 'unknown',
                         is_eager=is_eager, text=text[:50] if text else '')
                logger.info(f'SIP_DEBUG Transcribed utterance #{utterance_num}: {text}')
                # Capture how much of the in-flight TTS response actually played
                # BEFORE we cancel it, so we can truncate the persisted assistant
                # 'speak' text to roughly what the caller really heard (barge-in).
                spoken_seconds = 0.0
                try:
                    spoken_seconds = await service_manager.sip_response_spoken_seconds(context=ctx)
                except Exception as _e:
                    logger.debug(f'SIP_DEBUG could not read spoken_seconds: {_e}')
                res = await service_manager.cancel_and_wait(ctx.log_id, ctx.username)
                logger.info(f'SIP_DEBUG cancel result: {res}')
                # Rewrite the last assistant message to reflect only the speech
                # that was actually voiced before this barge-in.
                try:
                    # Always reconcile the persisted assistant 'speak' text with
                    # what was actually voiced. spoken_seconds == 0 is a VALID and
                    # important case: the response was barged-in/halted during the
                    # TTS warm-up before any audio reached the wire, so the entire
                    # speak text should be dropped (budget=0 -> drop). Guarding on
                    # '> 0' previously left the full untruncated text in the log.
                    sp = spoken_seconds if (spoken_seconds and spoken_seconds > 0) else 0.0
                    tr = await service_manager.truncate_last_assistant_speech(
                        spoken_seconds=sp, context=ctx)
                    logger.info(f'SIP_DEBUG truncate_last_assistant_speech: {tr} (spoken={sp:.2f}s)')
                except Exception as _e:
                    logger.warning(f'SIP_DEBUG truncate_last_assistant_speech failed: {_e}')
                session_manager = get_session_manager()
                session = await session_manager.get_session(ctx.log_id)
                if session:
                    session.resume_audio()
                else:
                    pass
                await service_manager.backend_user_message(message=text)
                logger.info(f'SIP_DEBUG Sending message to agent for session {ctx.log_id}')
                # Dead-air backstop: mark reply generation boundaries so an
                # un-voiced reply (speak text generated but 0 RTP frames) can be
                # re-delivered once the far end turn is over. No behavior change
                # unless MR_SIP_DEADAIR_BACKSTOP_ENABLED=1.
                _bot = getattr(session, 'baresip_bot', None) if session else None
                if _bot is not None and hasattr(_bot, 'note_reply_generation_start'):
                    try:
                        _bot.note_reply_generation_start()
                    except Exception:
                        pass
                await service_manager.send_message_to_agent(session_id=ctx.log_id, message=text, context=ctx)
                if _bot is not None and hasattr(_bot, 'note_reply_generation_done'):
                    try:
                        _gen_text = await service_manager.get_last_assistant_speech_text(context=ctx)
                    except Exception:
                        _gen_text = ''
                    try:
                        _bot.note_reply_generation_done(len(_gen_text or ''), _gen_text or '')
                    except Exception:
                        pass
            except Exception as e:
                logger.error(f'SIP_DEBUG Error processing utterance: {e}')
            finally:
                pass
        stt_config = {}
        if stt_provider in ['deepgram', 'deepgram_flux']:
            logger.info(f'{stt_provider} configuration prepared')
            if os.environ.get('DEEPGRAM_EOT_SECONDS', None) is not None:
                try:
                    eot = float(os.environ.get('DEEPGRAM_EOT_SECONDS'))
                    if eot > 0:
                        stt_config['eot_threshold'] = eot
                        logger.info(f'Using DEEPGRAM_EOT_SECONDS={eot}')
                    else:
                        pass
                except ValueError:
                    logger.warning(f'Invalid DEEPGRAM_EOT_SECONDS value')
                finally:
                    pass
            else:
                pass
            if os.environ.get('DEEPGRAM_EAGER_EOT_SECONDS', None) is not None:
                try:
                    eager_eot = float(os.environ.get('DEEPGRAM_EAGER_EOT_SECONDS'))
                    if eager_eot > 0:
                        stt_config['eager_eot_threshold'] = eager_eot
                        logger.info(f'Using DEEPGRAM_EAGER_EOT_SECONDS={eager_eot}')
                    else:
                        pass
                except ValueError:
                    logger.warning(f'Invalid DEEPGRAM_EAGER_EOT_SECONDS value')
                finally:
                    pass
            else:
                pass
            stt_config['keyterm'] = ['employee', 'employees', 'employment verification', 'manager', 'HR', 'date-of-birth']
        elif stt_provider == 'silero_cohere':
            logger.info('silero_cohere configuration prepared')
            for env_key, cfg_key in [
                ('SILERO_VAD_THRESHOLD',    'threshold'),
                ('SILERO_EAGER_SILENCE_MS', 'eager_silence_ms'),
                ('SILERO_FINAL_SILENCE_MS', 'final_silence_ms'),
                ('SILERO_MIN_SILENCE_MS',   'min_silence_duration_ms'),  # legacy compat
                ('SILERO_SPEECH_PAD_MS',    'speech_pad_ms'),
                ('COHERE_TRANSCRIBE_MODEL', 'cohere_model_id'),
                ('COHERE_TRANSCRIBE_LANGUAGE', 'language'),
                ('COHERE_MAX_UTTERANCE_S',  'max_utterance_duration_s'),
                ('COHERE_TRANSCRIBE_URL',   'cohere_transcribe_url'),
            ]:
                val = os.environ.get(env_key)
                if val is not None:
                    stt_config[cfg_key] = val
        elif stt_provider == 'umut':
            logger.info('umut (Kyutai streaming ASR + pause head) configuration prepared')
            for env_key, cfg_key in [
                ('UMUT_STT_URL', 'stt_url'),
                ('UMUT_API_KEY', 'api_key'),
                ('UMUT_END_THRESHOLD', 'end_threshold'),
                ('UMUT_SPEECH_THRESHOLD', 'speech_threshold'),
                ('UMUT_ASR_DELAY_SEC', 'delay_sec'),
                ('UMUT_QUEUE_FRAMES', 'queue_frames'),
                ('UMUT_VAD_INTERRUPTION', 'vad_interruption'),
            ]:
                val = os.environ.get(env_key)
                if val is not None:
                    stt_config[cfg_key] = val
        elif stt_provider == 'whisper_vad':
            stt_model_size = os.getenv('STT_MODEL_SIZE', 'small')
            stt_config['model_size'] = stt_model_size
            logger.info(f'Whisper VAD configuration prepared (model: {stt_model_size})')
        else:
            pass
        bot = MindRootSIPBotV2(user=sip_user, password=sip_password, gateway=sip_gateway, audio_dir=audio_dir, on_utterance_callback=on_utterance_callback, stt_provider=stt_provider, stt_config=stt_config, context=context, enable_recording=enable_recording, recording_dir=recording_dir, record_separate=record_separate)
        session_manager = get_session_manager()
        session = await session_manager.create_session(log_id=context.log_id, destination=destination, baresip_bot=bot)
        call_task = asyncio.create_task(bot.make_call(destination))
        logger.info(f'Waiting for call to be answered (timeout: {call_establish_timeout}s)...')
        try:
            await asyncio.wait_for(bot.call_answered.wait(), timeout=call_establish_timeout)
            session.is_active = True
            await session.start_audio_sender()
            logger.info(f'Call established to {destination}')
            return {'status': 'call_established', 'log_id': context.log_id, 'destination': destination, 'stt_provider': stt_provider, 'mode': 'pysip_v2', 'session_created_at': session.created_at.isoformat(), 'recording_enabled': enable_recording}
        except asyncio.TimeoutError:
            logger.error(f'Call to {destination} not answered within {call_establish_timeout}s')
            bot._aborted = True
            try:
                await bot._terminate_call(
                    f'Call setup timeout: not answered within {call_establish_timeout}s',
                    state=CallState.FAILED,
                )
            except Exception as e:
                logger.warning(f'Error terminating timed-out call to {destination}: {e}')
            await session_manager.end_session(context.log_id)
            if not call_task.done():
                call_task.cancel()
                try:
                    await call_task
                except asyncio.CancelledError:
                    pass
                finally:
                    pass
            else:
                pass
            return {'status': 'call_failed', 'log_id': context.log_id, 'destination': destination, 'error': 'Call not answered within timeout'}
        finally:
            pass
    except Exception as e:
        logger.error(f'Error in dial_service_v2: {e}')
        import traceback
        logger.error(traceback.format_exc())
        return {'status': 'error', 'log_id': context.log_id if context else None, 'destination': destination, 'error': str(e)}
    finally:
        pass

@service()
async def end_call_service_v2(context=None) -> Dict[str, Any]:
    """
    Service to terminate active V2 SIP call and cleanup resources.
    
    Args:
        context: MindRoot context (required for session identification)
    
    Returns:
        dict: Status information about the call termination
    """
    if not context or not context.log_id:
        return {'status': 'error', 'error': 'Context with log_id is required'}
    else:
        pass
    try:
        session_manager = get_session_manager()
        session = await session_manager.get_session(context.log_id)
        if session and session.baresip_bot:
            call_duration = None
            transcript = ''
            if session.baresip_bot.call_start_time:
                call_duration = (datetime.now() - session.baresip_bot.call_start_time).total_seconds()
            else:
                pass
            transcript = session.baresip_bot.get_transcript()
            await session.baresip_bot.hangup_call()
            await session_manager.end_session(context.log_id)
            logger.info(f'Successfully ended V2 SIP call for session {context.log_id}')
            return {'status': 'call_ended', 'log_id': context.log_id, 'call_duration_seconds': call_duration, 'transcript': transcript, 'mode': 'pysip_v2'}
        else:
            return {'status': 'no_active_call', 'log_id': context.log_id}
    except Exception as e:
        logger.error(f'Error in end_call_service_v2: {e}')
        import traceback
        logger.error(traceback.format_exc())
        return {'status': 'error', 'log_id': context.log_id if context else None, 'error': str(e)}
    finally:
        pass

@service()
async def sip_start_audio_response(context=None) -> bool:
    """Mark the start of one outbound AI/TTS audio response.

    TTS providers that know response boundaries should call this before the
    first sip_audio_out_chunk() for a response.  mr_sip keeps the implementation
    detail hidden from providers: currently this creates a fresh PySIP audio
    stream so PySIP's outgoing prebuffer starts at the beginning of the actual
    AI response.
    """
    if not context or not context.log_id:
        logger.warning('sip_start_audio_response called without context or log_id')
        return False
    try:
        session_manager = get_session_manager()
        session = await session_manager.get_session(context.log_id)
        if session and session.is_active:
            if hasattr(session, 'start_audio_response'):
                await session.start_audio_response()
                logger.debug(f'Started audio response for session {context.log_id}')
                return True
            if session.baresip_bot and hasattr(session.baresip_bot, 'start_tts_response'):
                return await session.baresip_bot.start_tts_response()
        logger.warning(f'No active SIP session found for audio response start: {context.log_id}')
        return False
    except Exception as e:
        logger.error(f'Error in sip_start_audio_response: {e}')
        return False

@service()
async def sip_end_audio_response(context=None) -> bool:
    """Mark the end of one outbound AI/TTS audio response.

    The end marker is ordered behind already queued audio chunks so PySIP can
    drain the response tail before returning to ordinary silence.
    """
    if not context or not context.log_id:
        logger.warning('sip_end_audio_response called without context or log_id')
        return False
    try:
        session_manager = get_session_manager()
        session = await session_manager.get_session(context.log_id)
        if session and session.is_active:
            if hasattr(session, 'end_audio_response'):
                await session.end_audio_response()
                logger.debug(f'Ended audio response for session {context.log_id}')
                return True
            if session.baresip_bot and hasattr(session.baresip_bot, 'end_tts_response'):
                return await session.baresip_bot.end_tts_response()
        logger.debug(f'No active SIP session found for audio response end: {context.log_id}')
        return False
    except Exception as e:
        logger.error(f'Error in sip_end_audio_response: {e}')
        return False

@service()
async def sip_audio_out_chunk(audio_chunk: bytes, timestamp=None, context=None) -> bool:
    """
    Service to route TTS audio chunks to active SIP call.
    
    This is called by TTS services (mr_eleven_stream, etc.) to send
    audio to the phone call.
    
    Args:
        audio_chunk: Raw audio data bytes
        timestamp: Optional timestamp for audio pacing
        context: MindRoot context (required for session identification)
    
    Returns:
        bool: True if audio was successfully queued, False otherwise
    """
    logger.debug(f'sip_audio_out_chunk called with {len(audio_chunk)} bytes')
    if not context or not context.log_id:
        logger.warning('sip_audio_out_chunk called without context or log_id')
        return False
    else:
        pass
    try:
        session_manager = get_session_manager()
        session = await session_manager.get_session(context.log_id)
        if session and session.is_active:
            if session.halt_audio_out:
                logger.debug('Audio halted - not outputting chunk')
                return False
            else:
                # Backward-compatible S2S safety: existing speech-to-speech
                # code paths call sip_audio_out_chunk() directly from OpenAI
                # output_audio.delta callbacks and may not call explicit
                # sip_start/end_audio_response services.  Do not make explicit
                # lifecycle mandatory here unless S2S is updated too.
                await session.send_audio(audio_chunk, timestamp=timestamp)
                logger.debug(f'Queued audio chunk for session {context.log_id}: {len(audio_chunk)} bytes')
                return True
        else:
            logger.warning(f'No active SIP session found for log_id {context.log_id}')
            return False
    except Exception as e:
        logger.error(f'Error in sip_audio_out_chunk: {e}')
        return False
    finally:
        pass

@service()
async def sip_clear_audio_queue(context=None) -> Dict[str, Any]:
    """
    Service to clear all queued audio for interruption.
    
    Called when user interruption is detected to immediately
    stop playing the current response.
    
    Args:
        context: MindRoot context (required for session identification)
    
    Returns:
        dict: Status information
    """
    if not context or not context.log_id:
        return {'status': 'error', 'error': 'Context with log_id is required'}
    else:
        pass
    try:
        session_manager = get_session_manager()
        session = await session_manager.get_session(context.log_id)
        if session and session.is_active:
            if session.baresip_bot:
                session.baresip_bot.clear_audio_queue()
            else:
                pass
            logger.info(f'Cleared audio queue for session {context.log_id}')
            return {'status': 'cleared', 'log_id': context.log_id}
        else:
            return {'status': 'no_active_session', 'log_id': context.log_id}
    except Exception as e:
        logger.error(f'Error in sip_clear_audio_queue: {e}')
        return {'status': 'error', 'log_id': context.log_id, 'error': str(e)}
    finally:
        pass

@service()
async def sip_halt_audio(context=None) -> Dict[str, Any]:
    """
    Halt audio output for the SIP session (for interruption).
    
    This is called when user interruption is detected to immediately
    stop sending audio. The halt persists until sip_resume_audio is called.
    
    Args:
        context: MindRoot context (required for session identification)
    
    Returns:
        dict: Status information
    """
    if not context or not context.log_id:
        return {'status': 'error', 'error': 'Context with log_id is required'}
    else:
        pass
    try:
        session_manager = get_session_manager()
        session = await session_manager.get_session(context.log_id)
        if session and session.is_active:
            session.halt_audio()
            logger.info(f'Audio HALTED for session {context.log_id}')
            return {'status': 'halted', 'log_id': context.log_id}
        else:
            return {'status': 'no_active_session', 'log_id': context.log_id}
    except Exception as e:
        logger.error(f'Error in sip_halt_audio: {e}')
        return {'status': 'error', 'log_id': context.log_id, 'error': str(e)}
    finally:
        pass

@service()
async def sip_resume_audio(context=None) -> bool:
    """
    Resume audio output for the SIP session.
    
    This is called when a new AI response starts to allow audio to flow again.
    
    Args:
        context: MindRoot context (required for session identification)
    
    Returns:
        bool: True if resumed, False otherwise
    """
    if not context or not context.log_id:
        logger.warning('sip_resume_audio called without context or log_id')
        return False
    else:
        pass
    try:
        session_manager = get_session_manager()
        session = await session_manager.get_session(context.log_id)
        if session and session.is_active:
            session.resume_audio()
            return True
        else:
            return False
    except Exception as e:
        logger.error(f'Error in sip_resume_audio: {e}')
        return False
    finally:
        pass

@service()
async def sip_is_audio_halted(context=None) -> bool:
    """
    Check if audio output is currently halted for the SIP session.
    
    Args:
        context: MindRoot context (required for session identification)
    
    Returns:
        bool: True if audio is halted, False otherwise
    """
    if not context or not context.log_id:
        return False
    else:
        pass
    try:
        session_manager = get_session_manager()
        session = await session_manager.get_session(context.log_id)
        if session and session.is_active:
            return session.halt_audio_out
        else:
            pass
        return False
    except Exception as e:
        logger.error(f'Error in sip_is_audio_halted: {e}')
        return False
    finally:
        pass

@service()
async def sip_response_spoken_seconds(context=None) -> float:
    """Return approximate seconds of the current/last outbound TTS response that
    were actually sent to the call before now.

    Because TTS plugins (mr_kyutai, mr_eleven_stream, ...) pace audio to real
    time before handing chunks to sip_audio_out_chunk, the volume of audio that
    has actually been dequeued+sent for this response is a good proxy for how
    much of it the caller actually heard. On a barge-in, mindroot uses this to
    truncate the persisted assistant 'speak' text to roughly what was voiced.

    Returns 0.0 if there is no active session.
    """
    if not context or not context.log_id:
        return 0.0
    try:
        session_manager = get_session_manager()
        session = await session_manager.get_session(context.log_id)
        if session and hasattr(session, 'played_seconds'):
            return session.played_seconds()
        return 0.0
    except Exception as e:
        logger.error(f'Error in sip_response_spoken_seconds: {e}')
        return 0.0



# Global incoming call listener instance
_incoming_listener = None

@service()
async def start_incoming_listener_service(agent_name: str = None, context=None):
    """
    Start the SIP incoming call listener.
    
    DEBUG: This service was called.
    
    Registers the SIP account with the provider and listens for
    incoming INVITEs. When a call arrives, creates a MindRoot
    chat session with the specified agent and wires audio/STT/TTS.
    
    Args:
        agent_name: Which MindRoot agent answers incoming calls.
                    Defaults to SIP_INCOMING_AGENT env var.
        context: MindRoot context (optional)
    
    Returns:
        dict: Status information
    
    Environment Variables:
        SIP_INCOMING_AGENT: Default agent for incoming calls
        SIP_INCOMING_DEFAULT_USER: User context for incoming sessions (default: system)
        SIP_GATEWAY: SIP gateway server
        SIP_USER: SIP username
        SIP_PASSWORD: SIP password
        SIP_CALLER_ID: Caller ID for registration
        STT_PROVIDER: STT provider to use
    """
    global _incoming_listener
    
    if _incoming_listener is not None:
        health = None
        try:
            if hasattr(_incoming_listener, 'health_info'):
                health = _incoming_listener.health_info()
        except Exception as e:
            health = {'healthy': False, 'error': str(e)}

        if health and health.get('healthy'):
            return {
                'status': 'already_running',
                'message': 'Incoming call listener is already active',
                'health': health,
            }

        logger.warning('[INCOMING-SVC] Existing incoming listener is stale/unhealthy; restarting it. health=%s', health)
        try:
            await _incoming_listener.stop()
        except Exception as e:
            logger.error('[INCOMING-SVC] Error stopping stale incoming listener before restart: %s', e)
        _incoming_listener = None
    
    sip_gateway = os.getenv('SIP_GATEWAY', 'no sip gateway')
    sip_user = os.getenv('SIP_USER', 'nouser')
    sip_password = os.getenv('SIP_PASSWORD', 'no sip password')
    caller_id = os.getenv('SIP_CALLER_ID', sip_user)
    agent = agent_name or os.getenv('SIP_INCOMING_AGENT', 'default')
    stt_provider = os.getenv('STT_PROVIDER', 'deepgram_flux')
    enable_recording = os.getenv('SIP_ENABLE_RECORDING', 'false').lower() == 'true'
    recording_dir = os.getenv('SIP_RECORDING_DIR', 'data/calls')
    record_separate = os.getenv('SIP_RECORD_SEPARATE', 'false').lower() == 'true'
    
    logger.info(f'[INCOMING-SVC] Starting incoming call listener for {sip_user}@{sip_gateway}')
    logger.info(f'[INCOMING-SVC] Agent: {agent}, STT: {stt_provider}')
    logger.info(f'[INCOMING-SVC] Caller ID: {caller_id}, Recording: {enable_recording}')
    
    try:
        _incoming_listener = MindRootSIPAccount(
            user=sip_user,
            password=sip_password,
            gateway=sip_gateway,
            agent_name=agent,
            caller_id=caller_id,
            stt_provider=stt_provider,
            enable_recording=enable_recording,
            recording_dir=recording_dir,
            record_separate=record_separate,
        )
        
        is_registered = await _incoming_listener.start()
        
        if is_registered:
            logger.info('[INCOMING-SVC] Incoming call listener started successfully')
            return {
                'status': 'started',
                'agent': agent,
                'gateway': sip_gateway,
                'user': sip_user,
                'caller_id': caller_id,
            }
        else:
            _incoming_listener = None
            logger.error('[INCOMING-SVC] SIP registration failed!')
            return {
                'status': 'failed',
                'error': 'SIP registration failed'
            }
            
    except Exception as e:
        logger.error(f'Error starting incoming call listener: {e}')
        _incoming_listener = None
        return {
            'status': 'error',
            'error': str(e)
        }

@service()
async def stop_incoming_listener_service(context=None):
    """
    Stop the SIP incoming call listener.
    
    DEBUG: This service was called.
    
    Unregisters from the SIP provider and stops listening for
    incoming calls. Active calls are NOT terminated.
    
    Returns:
        dict: Status information
    """
    global _incoming_listener
    
    if _incoming_listener is None:
        return {
            'status': 'not_running',
            'message': 'Incoming call listener is not active'
        }
    
    try:
        await _incoming_listener.stop()
        _incoming_listener = None
        logger.info('Incoming call listener stopped')
        return {
            'status': 'stopped',
            'message': 'Incoming call listener stopped'
        }
    except Exception as e:
        logger.error(f'Error stopping incoming call listener: {e}')
        return {
            'status': 'error',
            'error': str(e)
        }

@service()
async def get_incoming_listener_status(context=None):
    """
    Get the status of the incoming call listener.
    
    Returns:
        dict: Status information including whether listener is active
    """
    global _incoming_listener
    
    if _incoming_listener is None:
        return {
            'status': 'not_running',
            'is_active': False
        }
    
    health = None
    try:
        if hasattr(_incoming_listener, 'health_info'):
            health = _incoming_listener.health_info()
    except Exception as e:
        health = {'healthy': False, 'error': str(e)}

    return {
        'status': 'running' if (health is None or health.get('healthy')) else 'stale',
        'is_active': bool(health is None or health.get('healthy')),
        'agent': _incoming_listener.agent_name,
        'gateway': _incoming_listener.sip_server,
        'user': _incoming_listener.sip_username,
        'health': health,
    }

# Lock to prevent concurrent startup hook execution
_startup_lock = asyncio.Lock()

@hook()
async def startup(app=None, context=None):
    """Auto-start incoming call listener on plugin load if configured."""
    global _incoming_listener
    agent = os.getenv('SIP_INCOMING_AGENT')
    auto_start = os.getenv('SIP_INCOMING_AUTO_START', 'true').lower() in ('true', '1', 'yes', 'on')
    
    logger.info(f'[INCOMING] startup hook called. agent={agent}, auto_start={auto_start}')
    
    if agent and auto_start:
        logger.info(f'[INCOMING] Auto-starting incoming call listener (agent={agent})...')
        async with _startup_lock:
            # Double-check under lock in case concurrent call already started it.
            # If the object exists but the underlying PySIP receive/register
            # loop is stale, start_incoming_listener_service() will restart it.
            if _incoming_listener is not None:
                try:
                    health = _incoming_listener.health_info() if hasattr(_incoming_listener, 'health_info') else None
                except Exception as e:
                    health = {'healthy': False, 'error': str(e)}
                if health and health.get('healthy'):
                    logger.info('[INCOMING] Auto-start: listener already running and healthy (checked under lock)')
                    return
                logger.warning('[INCOMING] Auto-start found stale listener; restarting. health=%s', health)
            try:
                result = await start_incoming_listener_service(agent_name=agent)
                if result.get('status') == 'started':
                    logger.info(f'[INCOMING] Auto-start successful: {result}')
                elif result.get('status') == 'already_running':
                    logger.info(f'[INCOMING] Auto-start: listener already running')
                else:
                    logger.error(f'[INCOMING] Auto-start failed: {result}')
            except Exception as e:
                logger.error(f'[INCOMING] Auto-start error: {e}')
    else:
        if not agent:
            logger.info('[INCOMING] Auto-start skipped: SIP_INCOMING_AGENT not set')
        else:
            logger.info('[INCOMING] Auto-start disabled (set SIP_INCOMING_AUTO_START=true to enable)')

@hook()
async def quit(context=None):
    """Cleanup hook called when MindRoot is shutting down"""
    logger.info('MindRoot SIP plugin (V2) shutting down...')
    try:
        session_manager = get_session_manager()
        await session_manager.cleanup_all_sessions()
        logger.info('All SIP sessions cleaned up')
    except Exception as e:
        logger.error(f'Error during SIP plugin shutdown: {e}')
    finally:
        pass
    return {'status': 'sip_plugin_v2_shutdown_complete'}
logger.info('MindRoot SIP plugin V2 (PySIP + Deepgram) services loaded')
