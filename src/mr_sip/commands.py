#!/usr/bin/env python3
"""
MindRoot SIP Plugin - User Commands
"""

import logging
from lib.providers.commands import command
from .services import dial_service, end_call_service

logger = logging.getLogger(__name__)

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
        SIP_PASSWORD: SIP password (default: 3BM]ZEu:z4.]vXU)
        WHISPER_MODEL: Whisper model size (default: small)
        AUDIO_DIR: Audio recording directory (default: ~/.baresip)
    """
    try:
        if not destination:
            return "Error: Destination phone number or SIP URI is required"
            
        if not context or not context.log_id:
            return "Error: Valid MindRoot context is required for SIP calls"
        
        logger.info(f"Call command initiated to {destination} for session {context.log_id}")
        
        # Use the dial service to initiate the call
        result = await dial_service(destination=destination, context=context)
        
        if result["status"] == "call_established":
            return f"Call established to {destination}. Voice conversation is now active. Speak naturally and I will respond through the phone."
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
        
        # Use the end call service
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
