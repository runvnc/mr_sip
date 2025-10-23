#!/usr/bin/env python3
"""
MindRoot SIP Plugin - User Commands
"""

import os
import logging
import numpy as np
from lib.providers.commands import command, command_manager
from lib.chatcontext import get_context
from .services import dial_service, end_call_service
import nanoid
from .sip_manager import get_session_manager
import asyncio 
import traceback
import time

# Import V2 services if available
try:
    from .services_v2 import dial_service_v2, end_call_service_v2
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
        
        # strip punctuation from destination
        destination = ''.join(filter(str.isalnum, destination + '@'))
        # if it's just area code plus number, add default country code
        if destination.isdigit() and len(destination) == 10:
            destination = '1' + destination
        stt_provider = STT_PROVIDER
        logger.info(f"Using V2 implementation with STT provider: {stt_provider}")
        result = await dial_service_v2(destination=destination, context=context)
    
        if result["status"] == "call_established":
            msg = f"Call established to {destination}. Voice conversation is now active. Speak naturally and I will respond through the phone."
            
            # Add STT provider info if V2
            if result.get('stt_provider'):
                msg += f" (Using {result['stt_provider']} for transcription)"
            
            return None
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
        
        # Use the appropriate end call service based on version
        if USE_V2:
            result = await end_call_service_v2(context=context)
        else:
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

def generate_dtmf_tone(digit: str, duration: float = 0.1, sample_rate: int = 8000) -> np.ndarray:
    """
    Generate a DTMF tone for a single digit.
    
    DTMF uses two simultaneous tones (low and high frequency):
    
    Args:
        digit: Single DTMF digit (0-9, *, #)
        duration: Duration in seconds (default 0.1s = 100ms)
        sample_rate: Sample rate in Hz (default 8000 for phone audio)
    
    Returns:
        numpy array of float32 audio samples normalized to [-1, 1]
    """
    # DTMF frequency table
    dtmf_freqs = {
        '1': (697, 1209), '2': (697, 1336), '3': (697, 1477),
        '4': (770, 1209), '5': (770, 1336), '6': (770, 1477),
        '7': (852, 1209), '8': (852, 1336), '9': (852, 1477),
        '*': (941, 1209), '0': (941, 1336), '#': (941, 1477)
    }
    
    if digit not in dtmf_freqs:
        raise ValueError(f"Invalid DTMF digit: {digit}")
    
    low_freq, high_freq = dtmf_freqs[digit]
    
    # Generate time array
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    
    # Generate two sine waves and combine
    low_tone = np.sin(2 * np.pi * low_freq * t)
    high_tone = np.sin(2 * np.pi * high_freq * t)
    
    # Combine and normalize
    tone = (low_tone + high_tone) / 2.0
    
    # Apply envelope to avoid clicks (10ms fade in/out)
    fade_samples = int(0.01 * sample_rate)
    if fade_samples > 0:
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        tone[:fade_samples] *= fade_in
        tone[-fade_samples:] *= fade_out
    
    return tone.astype(np.float32)

def dtmf_to_ulaw(tone: np.ndarray) -> bytes:
    """
    Convert DTMF tone from float32 to μ-law encoded bytes.
    
    Args:
        tone: Float32 audio samples normalized to [-1, 1]
    
    Returns:
        μ-law encoded audio bytes
    """
    import audioop
    # Convert float32 to 16-bit PCM
    pcm = (tone * 32767).astype(np.int16).tobytes()
    # Convert PCM to μ-law
    return audioop.lin2ulaw(pcm, 2)

@command()
async def send_dtmf(digits: str, context=None) -> None:
    """
    Send DTMF tones during an active SIP call.
    
    DTMF (Dual-Tone Multi-Frequency) tones are used for phone menu navigation,
    entering PIN codes, or interacting with automated phone systems.
    
    Args:
        digits: String of DTMF digits to send (0-9, *, #)
                Can be a single digit or multiple digits
        context: MindRoot context (automatically provided)
    
    Returns:
        None: Command executes without waiting for acknowledgment
    
    Example:
        { "send_dtmf": { "digits": "1" } }
        { "send_dtmf": { "digits": "123#" } }
        { "send_dtmf": { "digits": "*9" } }
    """
    try:
        if not context or not context.log_id:
            logger.error("send_dtmf called without valid context")
            return
        
        if not digits:
            logger.error("send_dtmf called without digits")
            return
        
        # Validate digits
        valid_dtmf = set('0123456789*#')
        if not all(d in valid_dtmf for d in digits):
            logger.error(f"Invalid DTMF digits: {digits}")
            return
        
        session_manager = get_session_manager()
        session = await session_manager.get_session(context.log_id)
        
        if not session or not session.is_active:
            logger.warning(f"No active call for session {context.log_id}")
            return
        
        # Generate and send DTMF tones through the audio pipeline
        # This preserves the JACK audio setup unlike baresipy's send_dtmf
        logger.info(f"Generating DTMF tones for '{digits}'")
        
        for digit in digits:
            # Generate tone (100ms duration)
            tone = generate_dtmf_tone(digit, duration=0.1, sample_rate=8000)
            
            # Convert to μ-law format (same as TTS audio)
            ulaw_data = dtmf_to_ulaw(tone)
            
            # Send through the audio pipeline
            await session.send_audio(ulaw_data)
            
            # Add silence between digits (50ms)
            silence = np.zeros(int(0.05 * 8000), dtype=np.float32)
            silence_ulaw = dtmf_to_ulaw(silence)
            await session.send_audio(silence_ulaw)
            
            logger.debug(f"Sent DTMF tone for digit '{digit}'")
        
        logger.info(f"Sent DTMF digits '{digits}' for session {context.log_id}")
    except Exception as e:
        logger.error(f"Error in send_dtmf command: {e}")


@command()
async def wait(seconds:float, context=None) -> str:
    """
    Wait for a specified number of seconds during an active SIP call.
    
    This command pauses the MindRoot agent's processing for the given duration,
    allowing for timed interactions during a SIP call.
    You should use this if the transcribed text from the other party
    looks like it may be incomplete.
    
    Args:
        seconds: Number of seconds to wait (can be fractional)
        context: MindRoot context (automatically provided)
    
    Returns:
        str: Confirmation message after waiting
    
    Example:
        { "wait": { "seconds": 2.5 } }
    """
    try:
        if not context or not context.log_id:
            return "Error: Valid MindRoot context is required"
        
        if seconds <= 0:
            return "Error: Wait time must be greater than zero"
        
        logger.info(f"Waiting for {seconds} seconds during SIP call for session {context.log_id}")
        
        await asyncio.sleep(seconds)
        
        return f"Waited for {seconds} seconds."
        
    except Exception as e:
        logger.error(f"Error in wait command: {e}")
        return f"Error during wait: {str(e)}"

@command()
async def await_call_result(log_id: str, idle_timeout_seconds: int = 120, finish_timeout_seconds: int=20,context=None):
    """
    Wait for the call to end or inactivity timeout for the given log_id.
    This will return when: 
     
    - the chat session has returned a task_result

    - there is a CALL DISCONNECTED message in the log
      and finish_timeout_seconds has passed since the last change

    - idle_timeout_seconds has passed since the last change

    Example:

        { "await_call_result": { "log_id": "abc123", idle_timeout_seconds": 35, "finish_timeout_seconds": 5 } } 
    """
    try:
        call_context = await get_context(log_id, context.username)
        log = call_context.chat_log
        finished = False

        while not finished:
            await asyncio.sleep(1)
            idle = time.time() - log.last_modified
            if idle >= idle_timeout_seconds:
                logger.info(f"Call session {log_id} idle timeout reached ({idle_timeout_seconds}s)")
                finished = True
            commands = log.parsed_commands()
            for cmd in commands:
                if 'task_result' in cmd:
                    logger.info(f"Call session {log_id} received task_result")
                    return cmd['task_result']

            user_messages = [msg for msg in log.messages if msg.role == 'user']
            for msg in user_messages:
                if msg.content and isinstance(msg.content, list) and len(msg.content) > 0:
                    text = msg.content[0].get('text', '')
                    if "-- CALL DISCONNECTED --" in text:
                        logger.info(f"Call session {log_id} detected CALL DISCONNECTED message")
                        if idle >= finish_timeout_seconds:
                            logger.info(f"Call session {log_id} finish timeout reached ({finish_timeout_seconds}s) after disconnect")
                            finished = True

        log_dump = json.dumps(log.messages)
        return log_dump
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(f"Error in await_call_result: {e}\n\n{trace}")
        return f"Error awaiting call result: {str(e)} \n\n{trace}"

@command()
async def delegate_call_task(agent:str, phone_number:str, instructions: str, idle_timeout_seconds: int = 120,
                             finish_timeout_seconds: int=20, context=None):
    """
    Delegate a task to `agent` to call `phone_number` to accomplish task described in `instructions`.
    Wait for the the call to complete and return the task result from the call
    or the call session log if no task result.

    Example:

    { "delegate_call_task": { "agent": "CustomerService", "phone_number": "16822625850",
                              "instructions": "Call the customer and inform them about their order status." } }

    """
    log_id = nanoid.generate()
    await command_manager.delegate_task(instructions, agent, log_id=log_id, context=context)
    result = await await_call_result(log_id, idle_timeout_seconds=idle_timeout_seconds, 
                                     finish_timeout_seconds=finish_timeout_seconds, context=context)
    return result

