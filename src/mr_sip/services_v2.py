#!/usr/bin/env python3
"""
MindRoot SIP Plugin - Internal Services (V2 with STT Provider Support)

This version supports the new STT provider interface while maintaining
backward compatibility with the original implementation.
"""

import os
import asyncio
import logging
from typing import Dict, Any
from lib.providers.services import service, service_manager
from lib.providers.hooks import hook
from .sip_manager import get_session_manager
from .sip_client_v2 import MindRootSIPBotV2, setup_sndfile_module

logger = logging.getLogger(__name__)

# Configuration from environment
SIP_GATEWAY = os.getenv('SIP_GATEWAY', 'chicago4.voip.ms')
SIP_USER = os.getenv('SIP_USER', '498091')
SIP_PASSWORD = os.getenv('SIP_PASSWORD', '3BM]ZEu:z4.]vXU')
STT_PROVIDER = os.getenv('STT_PROVIDER', 'deepgram_flux')  # Default to deepgram_flux
STT_MODEL_SIZE = os.getenv('STT_MODEL_SIZE', 'small')  # For Whisper
DEEPGRAM_API_KEY = os.getenv('DEEPGRAM_API_KEY', '')  # For Deepgram
AUDIO_DIR = os.getenv('AUDIO_DIR', os.path.expanduser('.'))
REQUIRE_DEEPGRAM = os.getenv('REQUIRE_DEEPGRAM', 'true').lower() in ('true', '1', 'yes', 'on')
# Allow a longer ring timeout so Deepgram isn't started/given up before answer
CALL_ESTABLISH_TIMEOUT = int(os.getenv('SIP_CALL_ESTABLISH_TIMEOUT', '120'))

@service()
async def dial_service_v2(destination: str, context=None) -> Dict[str, Any]:
    """
    Service to initiate SIP calls using baresip with STT provider support.
    
    This is the V2 version that uses the abstract STT provider interface,
    allowing easy switching between Deepgram, Whisper, and future providers.
    
    Args:
        destination: Phone number or SIP URI to call
        context: MindRoot context (required for session linking)
    
    Returns:
        dict: Session information including log_id, destination, and status

    Environment Variables:
        STT_PROVIDER: 'deepgram_flux', 'deepgram', or 'whisper_vad' (default: 'deepgram_flux')
        DEEPGRAM_API_KEY: Required if using Deepgram
        STT_MODEL_SIZE: Whisper model size if using Whisper (default: 'small')
    """
    if not context or not context.log_id:
        raise ValueError("Context with log_id is required for SIP calls")
        
    logger.info(f"Initiating SIP call to {destination} for session {context.log_id}")
    logger.info(f"Using STT provider: {STT_PROVIDER}")
    
    # Enforce Deepgram requirement if configured
    if REQUIRE_DEEPGRAM:
        if STT_PROVIDER not in ['deepgram', 'deepgram_flux']:
            error_msg = (
                f"\n\n"
                f"{'='*80}\n"
                f"FATAL ERROR: Deepgram is required but STT_PROVIDER='{STT_PROVIDER}'\n"
                f"{'='*80}\n"
                f"Please set: export STT_PROVIDER=deepgram_flux (recommended) or deepgram\n"
                f"Or disable requirement: export REQUIRE_DEEPGRAM=false\n"
                f"{'='*80}\n"
            )
            logger.error(error_msg)
            import sys
            sys.exit(1)
        
        if not DEEPGRAM_API_KEY:
            error_msg = (
                f"\n\n"
                f"{'='*80}\n"
                f"FATAL ERROR: DEEPGRAM_API_KEY environment variable not set\n"
                f"{'='*80}\n"
                f"Deepgram is required but no API key was provided.\n"
                f"\n"
                f"To fix this:\n"
                f"1. Get an API key from https://deepgram.com/\n"
                f"2. Set it: export DEEPGRAM_API_KEY='your_key_here'\n"
                f"\n"
                f"Or to disable this requirement:\n"
                f"   export REQUIRE_DEEPGRAM=false\n"
                f"{'='*80}\n"
            )
            logger.error(error_msg)
            import sys
            sys.exit(1)
    
    try:
        # Verbose logging for Deepgram initialization
        if STT_PROVIDER in ['deepgram', 'deepgram_flux']:
            logger.info("\n" + "="*80)
            logger.info(f"INITIALIZING {STT_PROVIDER.upper()} STT PROVIDER")
            logger.info("="*80)
            logger.info(f"API Key: {DEEPGRAM_API_KEY[:10]}...{DEEPGRAM_API_KEY[-4:] if len(DEEPGRAM_API_KEY) > 14 else '[too short]'}")
            logger.info(f"Destination: {destination}")
            logger.info(f"Session: {context.log_id}")
            logger.info("="*80)
        
        # Check/setup sndfile module
        if not setup_sndfile_module():
            logger.warning("sndfile module setup failed, audio recording may not work")
        
        # Create utterance callback that sends messages to MindRoot agent
        async def on_utterance_callback(text: str, utterance_num: int, timestamp: float, ctx):
            """Callback for when complete utterances are transcribed"""
            try:
                logger.info(f"Transcribed utterance #{utterance_num}: {text}")
                
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
        
        # Prepare STT configuration
        stt_config = {}
        
        if STT_PROVIDER in ['deepgram', 'deepgram_flux']:
            # Don't put api_key in stt_config - it will be read from environment by factory
            # stt_config['api_key'] = DEEPGRAM_API_KEY  # Removed to avoid duplicate
            logger.info(f"{STT_PROVIDER} configuration prepared")
            # Skip test connection - will connect after call establishment
            logger.info(f"{STT_PROVIDER} will connect after call establishment")
                
        elif STT_PROVIDER == 'whisper_vad':
            stt_config['model_size'] = STT_MODEL_SIZE
            logger.info(f"Whisper VAD configuration prepared (model: {STT_MODEL_SIZE})")
        
        # Create baresip bot with MindRoot integration and STT provider
        bot = MindRootSIPBotV2(
            user=SIP_USER,
            password=SIP_PASSWORD,
            gateway=SIP_GATEWAY,
            audio_dir=AUDIO_DIR,
            on_utterance_callback=on_utterance_callback,
            stt_provider=STT_PROVIDER,
            stt_config=stt_config,
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
        max_wait = CALL_ESTABLISH_TIMEOUT  # seconds
        wait_count = 0
        while not bot.call_established and wait_count < max_wait:
            await asyncio.sleep(1)
            wait_count += 1
            
        if bot.call_established:
            session.is_active = True
            await session.start_audio_sender()
            logger.info(f"Call established to {destination}")
            
            return {
                "status": "call_established",
                "log_id": context.log_id,
                "destination": destination,
                "stt_provider": STT_PROVIDER,
                "session_created_at": session.created_at.isoformat()
            }
        else:
            # Call failed to establish
            await session_manager.end_session(context.log_id)
            logger.error(f"Failed to establish call to {destination} within {max_wait}s")
            
            return {
                "status": "call_failed",
                "log_id": context.log_id,
                "destination": destination,
                "error": "Call failed to establish within timeout"
            }
            
    except Exception as e:
        logger.error(f"Error in dial_service_v2: {e}")
        return {
            "status": "error",
            "log_id": context.log_id if context else None,
            "destination": destination,
            "error": str(e)
        }

# Note: sip_audio_out_chunk and end_call_service remain the same
# They work with both V1 and V2 implementations
