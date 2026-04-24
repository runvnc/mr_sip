"""
MindRoot SIP Plugin - Internal Services (V2 with PySIP + Deepgram STT)

This version uses PySIP for SIP/RTP handling instead of baresip+JACK.
Supports Deepgram Flux and other STT providers.
"""
import os
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any
from lib.providers.services import service, service_manager
from lib.providers.hooks import hook
from .sip_manager import get_session_manager
from .sip_client_v2 import MindRootSIPBotV2, setup_sndfile_module
from dotenv import load_dotenv
load_dotenv()
logger = logging.getLogger(__name__)

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
    is_local_provider = stt_provider in ('silero_cohere',)
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
                logger.info(f'SIP_DEBUG Transcribed utterance #{utterance_num}: {text}')
                res = await service_manager.cancel_and_wait(ctx.log_id, ctx.username)
                logger.info(f'SIP_DEBUG cancel result: {res}')
                session_manager = get_session_manager()
                session = await session_manager.get_session(ctx.log_id)
                if session:
                    session.resume_audio()
                else:
                    pass
                await service_manager.backend_user_message(message=text)
                logger.info(f'SIP_DEBUG Sending message to agent for session {ctx.log_id}')
                await service_manager.send_message_to_agent(session_id=ctx.log_id, message=text, context=ctx)
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
            bot._aborted = True
            await session_manager.end_session(context.log_id)
            logger.error(f'Call to {destination} not answered within {call_establish_timeout}s')
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