"""
MindRoot SIP Plugin - Speech-to-Speech Service Implementation (PySIP)

Provides dial_service and end_call_service for S2S mode using PySIP.
This implementation uses the PySIP-based S2S client for SIP call handling.
Audio output is handled by the SpeechToSpeechAgent calling sip_audio_out_chunk.
"""
import os
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any
from lib.providers.services import service
from .sip_manager import get_session_manager
from .sip_client_s2s import MindRootSIPBotS2S
from .pysip_process_wrapper import PySIPProcessWrapper
from .pysip_process_proxy import PySIPProcessProxy
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
async def dial_service(destination: str, context=None, enable_recording: bool=None, use_process_isolation: bool=True) -> Dict[str, Any]:
    """
    Service to initiate SIP calls in Speech-to-Speech mode using PySIP.
    
    This implementation uses the PySIP-based S2S client which handles both
    audio input and output. The SpeechToSpeechAgent manages the S2S session.
    
    Args:
        destination: Phone number or SIP URI to call
        context: MindRoot context (required for session linking)
        enable_recording: Override default recording setting (optional)
        use_process_isolation: Run PySIP in separate process (default: True)

    Returns:
        dict: Session information including log_id, destination, and status

    Environment Variables:
        SIP_GATEWAY: SIP gateway server (format: "host:port")
        SIP_USER: SIP username
        SIP_PASSWORD: SIP password
        SIP_PROVIDER: Must be 's2s' to use this implementation
        SIP_ENABLE_RECORDING: Enable call recording (default: false)
        SIP_RECORDING_DIR: Directory for recordings (default: recordings)
        SIP_RECORD_SEPARATE: Save separate incoming/outgoing files (default: false)
        SIP_USE_PROCESS_ISOLATION: Override process isolation setting
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
    audio_dir = os.getenv('AUDIO_DIR', os.path.expanduser('.'))
    call_establish_timeout = int(os.getenv('SIP_CALL_ESTABLISH_TIMEOUT', '120'))
    enable_recording_default = os.getenv('SIP_ENABLE_RECORDING', 'false').lower() == 'true'
    recording_dir = os.getenv('SIP_RECORDING_DIR', 'data/calls')
    record_separate = os.getenv('SIP_RECORD_SEPARATE', 'false').lower() == 'true'
    logger.info(f'Initiating PySIP call to {destination} for session {context.log_id} (S2S mode)')
    try:
        destination = ''.join(filter(str.isalnum, destination + '@'))
        if destination.isdigit() and len(destination) == 10:
            destination = '1' + destination
        else:
            pass
        env_isolation = os.getenv('SIP_USE_PROCESS_ISOLATION', '').lower()
        if env_isolation in ['true', 'false']:
            use_process_isolation = env_isolation == 'true'
            logger.info(f'Process isolation overridden by environment: {use_process_isolation}')
        else:
            pass
        record_call = enable_recording if enable_recording is not None else enable_recording_default
        if use_process_isolation:
            logger.info(f'Using PROCESS ISOLATION mode for call to {destination}')
            wrapper = PySIPProcessWrapper(context=context)
            bot = PySIPProcessProxy(wrapper=wrapper, context=context)
            await bot.make_call(destination=destination, user=sip_user, password=sip_password, gateway=sip_gateway, enable_recording=record_call, recording_dir=recording_dir, record_separate=record_separate)
        else:
            logger.info(f'Using DIRECT mode for call to {destination}')
            bot = MindRootSIPBotS2S(user=sip_user, password=sip_password, gateway=sip_gateway, audio_dir=audio_dir, context=context, enable_recording=record_call, recording_dir=recording_dir, record_separate=record_separate)
        session_manager = get_session_manager()
        session = await session_manager.create_session(log_id=context.log_id, destination=destination, baresip_bot=bot)
        if not use_process_isolation:
            call_task = asyncio.create_task(bot.make_call(destination))
            max_wait = call_establish_timeout
            logger.info(f'Waiting for call to be answered (timeout: {max_wait}s)...')
            try:
                await asyncio.wait_for(bot.call_answered.wait(), timeout=max_wait)
                bot.is_active = True
                bot.call_established = True
                bot.call_start_time = datetime.now()
            except asyncio.TimeoutError:
                await session_manager.end_session(context.log_id)
                logger.error(f'Call to {destination} not answered within {max_wait}s')
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
        else:
            pass
        session.is_active = True
        logger.info(f'S2S_DEBUG: Marking session {context.log_id} as active')
        await session.start_audio_sender()
        logger.info(f'S2S_DEBUG: Audio sender started for session {context.log_id}')
        logger.info(f"Call answered and ready to {destination} (mode: {('process_isolation' if use_process_isolation else 'direct')})")
        return {'status': 'call_established', 'log_id': context.log_id, 'destination': destination, 'mode': 's2s_pysip_isolated' if use_process_isolation else 's2s_pysip_direct', 'session_created_at': session.created_at.isoformat(), 'recording_enabled': record_call}
    except Exception as e:
        logger.error(f'Error in dial_service (PySIP S2S mode): {e}')
        import traceback
        logger.error(traceback.format_exc())
        return {'status': 'error', 'log_id': context.log_id if context else None, 'destination': destination, 'error': str(e)}
    finally:
        pass

@service()
async def end_call_service(context=None) -> Dict[str, Any]:
    """
    Service to terminate active SIP call in PySIP S2S mode.
    
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
            if session.baresip_bot.call_start_time:
                from datetime import datetime
                call_duration = (datetime.now() - session.baresip_bot.call_start_time).total_seconds()
            else:
                pass
            if hasattr(session.baresip_bot, 'stop_silence_monitor'):
                session.baresip_bot.stop_silence_monitor()
                logger.info(f'Stopped silence monitor for session {context.log_id}')
            else:
                pass
            await session.baresip_bot.hangup_call()
            await session_manager.end_session(context.log_id)
            logger.info(f'Successfully ended PySIP S2S SIP call for session {context.log_id}')
            return {'status': 'call_ended', 'log_id': context.log_id, 'call_duration_seconds': call_duration, 'mode': 's2s_pysip'}
        else:
            return {'status': 'no_active_call', 'log_id': context.log_id}
    except Exception as e:
        logger.error(f'Error in end_call_service (PySIP S2S mode): {e}')
        import traceback
        logger.error(traceback.format_exc())
        return {'status': 'error', 'log_id': context.log_id if context else None, 'error': str(e)}
    finally:
        pass

@service()
async def sip_clear_audio_queue(context=None) -> Dict[str, Any]:
    """
    Service to clear all queued audio for interruption.
    
    Called when OpenAI detects user interruption to immediately
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
                logger.warning(f'No bot found for session {context.log_id}')
            logger.info(f'Cleared audio queue for session {context.log_id}')
            return {'status': 'cleared', 'log_id': context.log_id}
        else:
            return {'status': 'no_active_session', 'log_id': context.log_id}
    except Exception as e:
        logger.error(f'Error in sip_clear_audio_queue: {e}')
        return {'status': 'error', 'log_id': context.log_id, 'error': str(e)}
    finally:
        pass