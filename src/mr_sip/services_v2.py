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
STT_PROVIDER = os.getenv('STT_PROVIDER', 'whisper_vad')  # 'deepgram' or 'whisper_vad'
STT_MODEL_SIZE = os.getenv('STT_MODEL_SIZE', 'small')  # For Whisper
DEEPGRAM_API_KEY = os.getenv('DEEPGRAM_API_KEY', '')  # For Deepgram
AUDIO_DIR = os.getenv('AUDIO_DIR', os.path.expanduser('.'))

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
        STT_PROVIDER: 'deepgram' or 'whisper_vad' (default: 'whisper_vad')
        DEEPGRAM_API_KEY: Required if using Deepgram
        STT_MODEL_SIZE: Whisper model size if using Whisper (default: 'small')
    """
    if not context or not context.log_id:
        raise ValueError("Context with log_id is required for SIP calls")
        
    logger.info(f"Initiating SIP call to {destination} for session {context.log_id}")
    logger.info(f"Using STT provider: {STT_PROVIDER}")
    
    try:
        # Check/setup sndfile module
        if not setup_sndfile_module():
            logger.warning("sndfile module setup failed, audio recording may not work")
        
        # Create utterance callback that sends messages to MindRoot agent
        async def on_utterance_callback(text: str, utterance_num: int, timestamp: float, ctx):
            """Callback for when complete utterances are transcribed"""
            try:
                logger.info(f"Transcribed utterance #{utterance_num}: {text}")
                
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
        
        if STT_PROVIDER == 'deepgram':
            if not DEEPGRAM_API_KEY:
                logger.error("DEEPGRAM_API_KEY not set but Deepgram provider selected")
                return {
                    "status": "error",
                    "log_id": context.log_id,
                    "destination": destination,
                    "error": "DEEPGRAM_API_KEY environment variable not set"
                }
            stt_config['api_key'] = DEEPGRAM_API_KEY
        elif STT_PROVIDER == 'whisper_vad':
            stt_config['model_size'] = STT_MODEL_SIZE
        
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
        max_wait = 30  # 30 seconds timeout
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
            logger.error(f"Failed to establish call to {destination}")
            
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
