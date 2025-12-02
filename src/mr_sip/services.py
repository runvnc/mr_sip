#!/usr/bin/env python3
"""
MindRoot SIP Plugin - Internal Services
"""

import os
import asyncio
import logging
from typing import Dict, Any
from lib.providers.services import service, service_manager
from lib.providers.hooks import hook
from .sip_manager import get_session_manager
from .sip_client import MindRootSIPBot, setup_sndfile_module
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


@service()
async def dial_service(destination: str, context=None) -> Dict[str, Any]:
    """
    Service to initiate SIP calls using baresip.
    
    Creates a SIP session linked to the MindRoot conversation context,
    sets up audio capture and transcription, and returns session information.
    
    Args:
        destination: Phone number or SIP URI to call
        context: MindRoot context (required for session linking)
    
    Returns:
        dict: Session information including log_id, destination, and status
    
    Example:
        session_info = await service_manager.dial_service(
            destination="16822625850",
            context=context
        )
    """
    if not context or not context.log_id:
        raise ValueError("Context with log_id is required for SIP calls")
        
    # Read environment variables inside the function where context is available
    # This allows context_environ to provide per-agent overrides
    sip_gateway = os.getenv('SIP_GATEWAY', 'no sip gateway')
    sip_user = os.getenv('SIP_USER', 'no sip user')
    sip_password = os.getenv('SIP_PASSWORD', 'no sip password')
    whisper_model = os.getenv('WHISPER_MODEL', 'small')
    audio_dir = os.getenv('AUDIO_DIR', os.path.expanduser('.'))

    logger.info(f"Initiating SIP call to {destination} for session {context.log_id}")
    
    try:
        # Check/setup sndfile module
        if not setup_sndfile_module():
            logger.warning("sndfile module setup failed, audio recording may not work")
        
        # Create utterance callback that sends messages to MindRoot agent
        async def on_utterance_callback(text: str, utterance_num: int, timestamp: float, ctx):
            """Callback for when complete utterances are transcribed"""
            try:
                logger.info(f"Transcribed utterance #{utterance_num}: {text}")
                
                await service_manager.cancel_and_wait(ctx.log_id, ctx.username)

                # Resume audio output - this is a NEW user message triggering a NEW response
                # This clears the halt flag so the new response can play
                session_manager = get_session_manager()
                session = await session_manager.get_session(ctx.log_id)
                if session:
                    session.resume_audio()

 
                # Update chat frontend with user message
                await service_manager.backend_user_message(
                    message=text
                )
                # Send transcribed text as user message to the agent
                await service_manager.send_message_to_agent(
                    session_id=ctx.log_id,
                    message=text,
                    context=ctx
                )
                
            except Exception as e:
                logger.error(f"Error processing utterance: {e}")
        
        # Create baresip bot with MindRoot integration
        bot = MindRootSIPBot(
            user=sip_user,
            password=sip_password,
            gateway=sip_gateway,
            audio_dir=audio_dir,
            on_utterance_callback=on_utterance_callback,
            model_size=whisper_model,
            context=context
        )
        
        # Wait for bot to be ready
        bot.wait_until_ready()
        
        # Create SIP session
        session_manager = get_session_manager()
        session = await session_manager.create_session(
            log_id=context.log_id,
            destination=destination,
            baresip_bot=bot
        )
        
        # Initiate the call
        logger.info(f"Calling {destination}...")
        bot.call(destination)
        
        # Wait for call to be established (with timeout)
        max_wait = 30  # 30 seconds timeout
        wait_count = 0
        while not bot.call_established and wait_count < max_wait:
            await asyncio.sleep(0.1)
            wait_count += 1
            
        if bot.call_established:
            session.is_active = True
            await session.start_audio_sender()
            logger.info(f"Call established to {destination}")
            
            return {
                "status": "call_established",
                "log_id": context.log_id,
                "destination": destination,
                "session_created_at": session.created_at.isoformat()
            }
        else:
            # Call failed to establish
            await session_manager.end_session(context.log_id)
            logger.error(f"Failed to establish call to {destination}")
            
            return {
                "status": "call_failed",
                "log_id": context.log_id,
                "destination": destination,
                "error": "Call failed to establish within timeout"
            }
            
    except Exception as e:
        logger.error(f"Error in dial_service: {e}")
        return {
            "status": "error",
            "log_id": context.log_id if context else None,
            "destination": destination,
            "error": str(e)
        }

@service()
async def sip_audio_out_chunk(audio_chunk: bytes, timestamp=None, context=None) -> bool:
    """
    Service to route TTS audio chunks to active SIP call.
    
    This is the critical service called by mr_eleven_stream's speak() command.
    It uses context.log_id to identify the target SIP session and routes
    audio chunks to the active call.
    
    Args:
        audio_chunk: Raw audio data bytes (typically ulaw_8000 format)
        timestamp: Optional timestamp when this audio should play (from AudioPacer)
        context: MindRoot context (required for session identification)
    
    Returns:
        bool: True if audio was successfully queued, False otherwise
    
    Example:
        # This is called automatically by mr_eleven_stream
        success = await service_manager.sip_audio_out_chunk(
            audio_chunk=chunk_data,
            context=context
        )
    """
    logger.debug(f"sip_audio_out_chunk called with {len(audio_chunk)} bytes")
    if not context or not context.log_id:
        logger.warning("sip_audio_out_chunk called without context or log_id")
        logger.warning("sip_audio_out_chunk called without context or log_id")
        logger.warning("sip_audio_out_chunk called without context or log_id")
        logger.warning("sip_audio_out_chunk called without context or log_id")
        logger.warning("sip_audio_out_chunk called without context or log_id")
    
        return False
        
    try:
        session_manager = get_session_manager()
        session = await session_manager.get_session(context.log_id)
        
        if session and session.is_active:
            if session.halt_audio_out:
                logger.debug(f"SIP DEBUG AUDIO HALTED NOT OUTPUTTING CHUNK")
 
                # REMOVED FOR BARGE-IN FIX:                 session.halt_audio_out = False
                return False
            else:
                await session.send_audio(audio_chunk, timestamp=timestamp)
                logger.debug(f"SIP DEBUG sip_audio_out_chunk active. Queued audio chunk for session {context.log_id}: {len(audio_chunk)} bytes")
 
                logger.debug(f"NOT HALTED. Queued audio chunk for session {context.log_id}: {len(audio_chunk)} bytes")
 
                return True
        else:
            logger.warning(f"No active SIP session found for log_id {context.log_id}")
            return False
            
    except Exception as e:
        logger.error(f"Error in sip_audio_out_chunk: {e}")
        return False

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
        return {
            "status": "error",
            "error": "Context with log_id is required"
        }
    
    try:
        session_manager = get_session_manager()
        session = await session_manager.get_session(context.log_id)
        
        if session and session.is_active:
            session.halt_audio()
            logger.info(f"Audio HALTED for session {context.log_id}")
            return {
                "status": "halted",
                "log_id": context.log_id
            }
        else:
            return {
                "status": "no_active_session",
                "log_id": context.log_id
            }
    except Exception as e:
        logger.error(f"Error in sip_halt_audio: {e}")
        return {
            "status": "error",
            "log_id": context.log_id,
            "error": str(e)
        }


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
        logger.warning("sip_resume_audio called without context or log_id")
        return False
    
    try:
        session_manager = get_session_manager()
        session = await session_manager.get_session(context.log_id)
        
        if session and session.is_active:
            session.resume_audio()
            return True
        else:
            return False
    except Exception as e:
        logger.error(f"Error in sip_resume_audio: {e}")
        return False


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
    
    try:
        session_manager = get_session_manager()
        session = await session_manager.get_session(context.log_id)
        
        if session and session.is_active:
            return session.halt_audio_out
        return False
    except Exception as e:
        logger.error(f"Error in sip_is_audio_halted: {e}")
        return False


@service()
async def end_call_service(context=None) -> Dict[str, Any]:
    """
    Service to terminate active SIP call and cleanup resources.
    
    Args:
        context: MindRoot context (required for session identification)
    
    Returns:
        dict: Status information about the call termination
    
    Example:
        result = await service_manager.end_call_service(context=context)
    """
    if not context or not context.log_id:
        return {
            "status": "error",
            "error": "Context with log_id is required"
        }
        
    try:
        session_manager = get_session_manager()
        session = await session_manager.get_session(context.log_id)
        
        if session:
            # Get call duration and transcript before ending
            call_duration = None
            transcript = ""
            
            if session.baresip_bot:
                if session.baresip_bot.call_start_time:
                    from datetime import datetime
                    call_duration = (datetime.now() - session.baresip_bot.call_start_time).total_seconds()
                transcript = session.baresip_bot.get_transcript()
            
            # End the session
            success = await session_manager.end_session(context.log_id)
            
            if success:
                logger.info(f"Successfully ended SIP call for session {context.log_id}")
                return {
                    "status": "call_ended",
                    "log_id": context.log_id,
                    "call_duration_seconds": call_duration,
                    "transcript": transcript
                }
            else:
                return {
                    "status": "error",
                    "log_id": context.log_id,
                    "error": "Failed to end session"
                }
        else:
            return {
                "status": "no_active_call",
                "log_id": context.log_id
            }
            
    except Exception as e:
        logger.error(f"Error in end_call_service: {e}")
        return {
            "status": "error",
            "log_id": context.log_id if context else None,
            "error": str(e)
        }

@hook()
async def quit(context=None):
    """Cleanup hook called when MindRoot is shutting down"""
    logger.info("MindRoot SIP plugin shutting down...")
    
    try:
        # Cleanup all active SIP sessions
        session_manager = get_session_manager()
        await session_manager.cleanup_all_sessions()
        logger.info("All SIP sessions cleaned up")
        
    except Exception as e:
        logger.error(f"Error during SIP plugin shutdown: {e}")
    
    return {"status": "sip_plugin_shutdown_complete"}

# Service registration verification  
logger.info("MindRoot SIP plugin services loaded")
