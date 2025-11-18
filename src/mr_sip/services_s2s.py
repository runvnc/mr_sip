#!/usr/bin/env python3
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

# Configuration from environment
SIP_GATEWAY = os.getenv('SIP_GATEWAY', 'no sip gateway')
SIP_USER = os.getenv('SIP_USER', 'nouser')
SIP_PASSWORD = os.getenv('SIP_PASSWORD', 'no sip password')
AUDIO_DIR = os.getenv('AUDIO_DIR', os.path.expanduser('.'))
CALL_ESTABLISH_TIMEOUT = int(os.getenv('SIP_CALL_ESTABLISH_TIMEOUT', '120'))
# Recording configuration
ENABLE_RECORDING = os.getenv('SIP_ENABLE_RECORDING', 'false').lower() == 'true'
RECORDING_DIR = os.getenv('SIP_RECORDING_DIR', 'data/calls')
RECORD_SEPARATE = os.getenv('SIP_RECORD_SEPARATE', 'false').lower() == 'true'

@service()
async def dial_service(destination: str, context=None, enable_recording: bool = None,
                      use_process_isolation: bool = True) -> Dict[str, Any]:
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
        raise ValueError("Context with log_id is required for SIP calls")
        
    print(f"[DIAL_SERVICE] Starting dial to {destination} for session {context.log_id}")
    logger.info(f"Initiating PySIP call to {destination} for session {context.log_id} (S2S mode)")
    
    try:
        # Strip punctuation from destination
        destination = ''.join(filter(str.isalnum, destination + '@'))
        # If it's just area code plus number, add default country code
        if destination.isdigit() and len(destination) == 10:
            destination = '1' + destination
        
        # Check environment variable override for process isolation
        env_isolation = os.getenv('SIP_USE_PROCESS_ISOLATION', '').lower()
        if env_isolation in ['true', 'false']:
            use_process_isolation = env_isolation == 'true'
            logger.info(f"Process isolation overridden by environment: {use_process_isolation}")
        print(f"[DIAL_SERVICE] Process isolation: {use_process_isolation}")
        
        # Determine recording setting  
        record_call = enable_recording if enable_recording is not None else ENABLE_RECORDING
        
        if use_process_isolation:
            logger.info(f"Using PROCESS ISOLATION mode for call to {destination}")
            
            print(f"[DIAL_SERVICE] Creating process wrapper...")
            # Create process wrapper
            wrapper = PySIPProcessWrapper(context=context)
            
            # Create proxy that will manage the subprocess
            bot = PySIPProcessProxy(wrapper=wrapper, context=context)
            
            # Start the call via proxy (this spawns subprocess)
            print(f"[DIAL_SERVICE] Calling bot.make_call()...")
            await bot.make_call(
                destination=destination,
                user=SIP_USER,
                password=SIP_PASSWORD,
                gateway=SIP_GATEWAY,
                enable_recording=record_call,
                recording_dir=RECORDING_DIR,
                record_separate=RECORD_SEPARATE
            )
            print(f"[DIAL_SERVICE] bot.make_call() returned successfully")
            
        else:
            logger.info(f"Using DIRECT mode for call to {destination}")
            
            print(f"[DIAL_SERVICE] Creating bot directly...")
            # Create bot directly (original behavior)
            bot = MindRootSIPBotS2S(
                user=SIP_USER,
                password=SIP_PASSWORD,
                gateway=SIP_GATEWAY,
                audio_dir=AUDIO_DIR,
                context=context,
                enable_recording=record_call,
                recording_dir=RECORDING_DIR,
                record_separate=RECORD_SEPARATE
            )
            print(f"[DIAL_SERVICE] Bot created, creating session...")
        
        session_manager = get_session_manager()
        session = await session_manager.create_session(
            log_id=context.log_id,
            destination=destination,
            baresip_bot=bot  # Keep parameter name for compatibility
        )
        print(f"[DIAL_SERVICE] Session created")
        
        if not use_process_isolation:
            # Only need to do this for direct mode
            # In process isolation mode, the proxy already started the call
            
            # Start the call in a task so we can monitor it
            call_task = asyncio.create_task(bot.make_call(destination))
            
            # Wait for call to be answered and RTP ready (with timeout)
            max_wait = CALL_ESTABLISH_TIMEOUT
            print(f"[DIAL_SERVICE] Starting call task and waiting up to {max_wait}s...")
            
            logger.info(f"Waiting for call to be answered (timeout: {max_wait}s)...")
            
            try:
                # Wait for the call_answered event (set when first RTP frame received)
                await asyncio.wait_for(bot.call_answered.wait(), timeout=max_wait)
                
                print(f"[DIAL_SERVICE] Call answered event received!")
                # Call is answered and RTP is ready!
                bot.is_active = True
                bot.call_established = True
                bot.call_start_time = datetime.now()
                
            except asyncio.TimeoutError:
                # Call not answered in time
                print(f"[DIAL_SERVICE] TIMEOUT waiting for call to be answered")
                await session_manager.end_session(context.log_id)
                logger.error(f"Call to {destination} not answered within {max_wait}s")
                
                # Cancel the call task if still running
                if not call_task.done():
                    call_task.cancel()
                    try:
                        await call_task
                    except asyncio.CancelledError:
                        pass
                
                return {
                    "status": "call_failed",
                    "log_id": context.log_id,
                    "destination": destination,
                    "error": "Call not answered within timeout"
                }
        
        print(f"[DIAL_SERVICE] Marking session as active and starting audio sender...")
        # Mark session as active and start audio sender
        session.is_active = True
        logger.info(f"S2S_DEBUG: Marking session {context.log_id} as active")
        
        await session.start_audio_sender()
        logger.info(f"S2S_DEBUG: Audio sender started for session {context.log_id}")
        logger.info(f"Call answered and ready to {destination} (mode: {'process_isolation' if use_process_isolation else 'direct'})")
        print(f"[DIAL_SERVICE] Returning success")
        
        return {
            "status": "call_established",
            "log_id": context.log_id,
            "destination": destination,
            "mode": "s2s_pysip_isolated" if use_process_isolation else "s2s_pysip_direct",
            "session_created_at": session.created_at.isoformat(),
            "recording_enabled": record_call
        }
            
    except Exception as e:
        logger.error(f"Error in dial_service (PySIP S2S mode): {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "log_id": context.log_id if context else None,
            "destination": destination,
            "error": str(e)
        }


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
        return {
            "status": "error",
            "error": "Context with log_id is required"
        }
        
    try:
        session_manager = get_session_manager()
        session = await session_manager.get_session(context.log_id)
        
        if session and session.baresip_bot:
            # Get call duration
            call_duration = None
            if session.baresip_bot.call_start_time:
                from datetime import datetime
                call_duration = (datetime.now() - session.baresip_bot.call_start_time).total_seconds()
            
            # Hangup the call (this triggers cleanup)
            await session.baresip_bot.hangup_call()
            
            # Clean up the session from the manager
            await session_manager.end_session(context.log_id)
            
            logger.info(f"Successfully ended PySIP S2S SIP call for session {context.log_id}")
            return {
                "status": "call_ended",
                "log_id": context.log_id,
                "call_duration_seconds": call_duration,
                "mode": "s2s_pysip"
            }
        else:
            return {
                "status": "no_active_call",
                "log_id": context.log_id
            }
            
    except Exception as e:
        logger.error(f"Error in end_call_service (PySIP S2S mode): {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "log_id": context.log_id if context else None,
            "error": str(e)
        }

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
        return {
            "status": "error",
            "error": "Context with log_id is required"
        }
    
    try:
        session_manager = get_session_manager()
        session = await session_manager.get_session(context.log_id)
        
        if session and session.is_active:
            # Call clear_audio_queue on the bot, not the session
            if session.baresip_bot:
                session.baresip_bot.clear_audio_queue()
            else:
                logger.warning(f"No bot found for session {context.log_id}")
            logger.info(f"Cleared audio queue for session {context.log_id}")
            return {
                "status": "cleared",
                "log_id": context.log_id
            }
        else:
            return {
                "status": "no_active_session",
                "log_id": context.log_id
            }
    except Exception as e:
        logger.error(f"Error in sip_clear_audio_queue: {e}")
        return {
            "status": "error",
            "log_id": context.log_id,
            "error": str(e)
        }

# Note: sip_audio_out_chunk service is reused from services.py
# It already handles routing audio to the active SIP session via session manager
