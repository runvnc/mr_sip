#!/usr/bin/env python3
"""
MindRoot SIP Plugin - User Commands
"""

import os
import logging
from lib.providers.commands import command
from .services import dial_service, end_call_service
from .sip_manager import get_session_manager

# Import V2 services if available
try:
    from .services_v2 import dial_service_v2
    V2_AVAILABLE = True
except ImportError:
    V2_AVAILABLE = False

# Import configuration
REQUIRE_DEEPGRAM = os.getenv('REQUIRE_DEEPGRAM', 'true').lower() in ('true', '1', 'yes', 'on')
STT_PROVIDER = os.getenv('STT_PROVIDER', 'deepgram' if REQUIRE_DEEPGRAM else 'whisper_vad')

logger = logging.getLogger(__name__)

# Check if V2 should be used (based on environment variable)
USE_V2 = V2_AVAILABLE and os.getenv('SIP_USE_V2', 'true').lower() in ('true', '1', 'yes', 'on')

# Log configuration on module load
logger.info(f"SIP Plugin Configuration: V2={'enabled' if USE_V2 else 'disabled'}, STT_PROVIDER={STT_PROVIDER}, REQUIRE_DEEPGRAM={REQUIRE_DEEPGRAM}")

@command()
async def call(destination: str, context=None) -> str:
    """
    Initiate a SIP phone call to the specified destination.
    
    This command creates a new SIP session linked to the current MindRoot
    conversation context, enabling voice conversations with the AI agent.
    
    Args:
        destination: Phone number or SIP URI to call (e.g., "16822625850")
        context: MindRoot context (automatically provided)
    
    Returns:
        str: Status message about the call initiation
    
    Example:
        { "call": { "destination": "16822625850" } }
        { "call": { "destination": "sip:user@domain.com" } }
    
    Environment Variables:
        SIP_GATEWAY: SIP gateway server (default: chicago4.voip.ms)
        SIP_USER: SIP username (default: 498091)
        SIP_PASSWORD: SIP password
        
        # V2 STT Provider Configuration (used if SIP_USE_V2=true)
        SIP_USE_V2: Use V2 implementation with STT providers (default: true)
        STT_PROVIDER: 'deepgram' or 'whisper_vad' (default: whisper_vad)
        DEEPGRAM_API_KEY: Required if STT_PROVIDER=deepgram
        STT_MODEL_SIZE: Whisper model size if STT_PROVIDER=whisper_vad (default: small)
        
        # V1 Configuration (used if SIP_USE_V2=false)
        WHISPER_MODEL: Whisper model size (default: small)
        AUDIO_DIR: Audio recording directory (default: ~/.baresip)
    """
    try:
        if not destination:
            return "Error: Destination phone number or SIP URI is required"
            
        if not context or not context.log_id:
            return "Error: Valid MindRoot context is required for SIP calls"
        
        logger.info(f"Call command initiated to {destination} for session {context.log_id}")
        
        # Use V2 if available and enabled
        if USE_V2:
            stt_provider = STT_PROVIDER
            logger.info(f"Using V2 implementation with STT provider: {stt_provider}")
            result = await dial_service_v2(destination=destination, context=context)
        else:
            if USE_V2 and not V2_AVAILABLE:
                logger.warning("V2 requested but not available, falling back to V1")
            logger.info("Using V1 implementation")
            result = await dial_service(destination=destination, context=context)
        
        if result["status"] == "call_established":
            msg = f"Call established to {destination}. Voice conversation is now active. Speak naturally and I will respond through the phone."
            
            # Add STT provider info if V2
            if result.get('stt_provider'):
                msg += f" (Using {result['stt_provider']} for transcription)"
            
            return msg
        elif result["status"] == "call_failed":
            return f"Failed to establish call to {destination}: {result.get('error', 'Unknown error')}"
        else:
            return f"Call initiation error: {result.get('error', 'Unknown error')}"
            
    except Exception as e:
        logger.error(f"Error in call command: {e}")
        return f"Error initiating call: {str(e)}"

@command()
async def hangup(context=None) -> str:
    """
    Terminate the current SIP phone call.
    
    This command ends the active SIP call associated with the current
    MindRoot conversation context and provides a summary of the call.
    
    Args:
        context: MindRoot context (automatically provided)
    
    Returns:
        str: Status message about the call termination and summary
    
    Example:
        { "hangup": {} }
    """
    try:
        if not context or not context.log_id:
            return "Error: Valid MindRoot context is required"
        
        logger.info(f"Hangup command initiated for session {context.log_id}")
        
        # Use the end call service (works with both V1 and V2)
        result = await end_call_service(context=context)
        
        if result["status"] == "call_ended":
            duration = result.get("call_duration_seconds")
            transcript = result.get("transcript", "")
            
            summary = f"Call ended successfully."
            if duration:
                summary += f" Duration: {duration:.1f} seconds."
            if transcript:
                summary += f" Transcript captured: {len(transcript.split())} words."
            
            return summary
        elif result["status"] == "no_active_call":
            return "No active call to hang up."
        else:
            return f"Error ending call: {result.get('error', 'Unknown error')}"
            
    except Exception as e:
        logger.error(f"Error in hangup command: {e}")
        return f"Error hanging up call: {str(e)}"

@command()
async def send_dtmf(digits: str, context=None) -> str:
    """
    Send DTMF tones during an active SIP call.
    
    DTMF (Dual-Tone Multi-Frequency) tones are used for phone menu navigation,
    entering PIN codes, or interacting with automated phone systems.
    
    Args:
        digits: String of DTMF digits to send (0-9, *, #)
                Can be a single digit or multiple digits
        context: MindRoot context (automatically provided)
    
    Returns:
        str: Status message about the DTMF transmission
    
    Example:
        { "send_dtmf": { "digits": "1" } }
        { "send_dtmf": { "digits": "123#" } }
        { "send_dtmf": { "digits": "*9" } }
    """
    try:
        if not context or not context.log_id:
            return "Error: Valid MindRoot context is required"
        
        if not digits:
            return "Error: DTMF digits are required"
        
        # Validate digits
        valid_dtmf = set('0123456789*#')
        if not all(d in valid_dtmf for d in digits):
            return f"Error: Invalid DTMF digits. Only 0-9, *, # are allowed. Got: {digits}"
        
        session_manager = get_session_manager()
        session = await session_manager.get_session(context.log_id)
        
        if session and session.is_active and session.baresip_bot:
            session.baresip_bot.send_dtmf(digits)
            logger.info(f"Sent DTMF digits '{digits}' for session {context.log_id}")
            return f"DTMF tones sent: {digits}"
        else:
            return "Error: No active call to send DTMF tones"
    except Exception as e:
        logger.error(f"Error in send_dtmf command: {e}")
        return f"Error sending DTMF: {str(e)}"
