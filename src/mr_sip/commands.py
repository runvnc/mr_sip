#!/usr/bin/env python3
"""
MindRoot SIP Plugin - User Commands
"""

print("mr_sip trying to load commands")

import os
import logging
import numpy as np
from lib.providers.commands import command, command_manager
from lib.chatcontext import get_context
from lib.chatlog import ChatLog
from .services import dial_service, end_call_service
import nanoid
from .sip_manager import get_session_manager
import asyncio 
from lib.providers.services import service_manager
import traceback
import time
import json
import time as time_module

# Lazy imports for S2S and V2 services - will be imported when needed
_s2s_services = None
_v2_services = None

print("mr_sip commands imports done")

def _get_sip_config(context=None):
    """Get SIP configuration from context or environment."""
    sip_provider = os.getenv('SIP_PROVIDER', 'deepgram').lower()
    require_deepgram = os.getenv('REQUIRE_DEEPGRAM', 'true').lower() in ('true', '1', 'yes', 'on')
    stt_provider = os.getenv('STT_PROVIDER', 'deepgram' if require_deepgram else 'whisper_vad')
    return sip_provider, require_deepgram, stt_provider

def _get_s2s_services():
    """Lazy load S2S services."""
    global _s2s_services
    if _s2s_services is None:
        try:
            from .services_s2s import dial_service as dial_service_s2s, end_call_service as end_call_service_s2s
            _s2s_services = (dial_service_s2s, end_call_service_s2s, True)
        except ImportError:
            _s2s_services = (None, None, False)
    return _s2s_services

def _get_v2_services():
    """Lazy load V2 services."""
    global _v2_services
    if _v2_services is None:
        try:
            from .services_v2 import dial_service_v2, end_call_service_v2
            _v2_services = (dial_service_v2, end_call_service_v2, True)
        except ImportError:
            _v2_services = (None, None, False)
    return _v2_services

# Check S2S availability at module load for logging purposes only
def _check_s2s_available():
    try:
        from .services_s2s import dial_service as dial_service_s2s, end_call_service as end_call_service_s2s
        return True
    except ImportError:
        return False

# Import configuration
logger = logging.getLogger(__name__)
logger.info("Commands module loaded - SIP config will be read from context/environment at runtime")

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
        SIP_GATEWAY: SIP gateway server
        SIP_USER: SIP username
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
        
        # Get config at runtime from context
        sip_provider, require_deepgram, stt_provider = _get_sip_config(context)
        logger.info(f"SIP config: provider={sip_provider}, stt={stt_provider}")
        
        # strip punctuation from destination
        destination = ''.join(filter(str.isalnum, destination + '@'))
        # if it's just area code plus number, add default country code
        if destination.isdigit() and len(destination) == 10:
            destination = '1' + destination
        
        # Use the appropriate dial service based on SIP_PROVIDER
        dial_service_s2s, end_call_service_s2s, s2s_available = _get_s2s_services()
        if sip_provider == 's2s' and s2s_available:
            logger.info(f"Using S2S implementation for call to {destination}")
            
            try:
                result = await asyncio.wait_for(
                    dial_service_s2s(destination=destination, context=context),
                    timeout=60.0
                )
            except asyncio.TimeoutError:
                logger.error(f"S2S dial service timed out after 60 seconds for destination {destination}")
                return f"Call initiation timed out after 60 seconds. The dial service did not respond."
        else:
            dial_service_v2, end_call_service_v2, v2_available = _get_v2_services()
            # Use V2 implementation
            logger.info(f"Using V2 implementation with STT provider: {stt_provider}")
            
            try:
                result = await asyncio.wait_for(
                    dial_service_v2(destination=destination, context=context),
                    timeout=60.0
                )
            except asyncio.TimeoutError:
                logger.error(f"Dial service timed out after 30 seconds for destination {destination}")
                return f"Call initiation timed out after 30 seconds. The dial service did not respond."
        
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
        
        # Get config at runtime from context
        sip_provider, require_deepgram, stt_provider = _get_sip_config(context)
        
        # Use the appropriate end call service based on version
        dial_service_s2s, end_call_service_s2s, s2s_available = _get_s2s_services()
        dial_service_v2, end_call_service_v2, v2_available = _get_v2_services()
        
        if sip_provider == 's2s' and s2s_available:
            result = await end_call_service_s2s(context=context)
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
        
        # Get current time as base timestamp for proper recording placement
        # Each digit's audio will be offset from this base time
        base_timestamp = time_module.perf_counter()
        
        for digit in digits:
            # Generate tone (100ms duration)
            tone = generate_dtmf_tone(digit, duration=0.1, sample_rate=8000)
            
            # Convert to μ-law format (same as TTS audio)
            ulaw_data = dtmf_to_ulaw(tone)
            
            # Send through the audio pipeline with timestamp for proper recording
            await session.send_audio(ulaw_data, timestamp=base_timestamp)
            
            # Advance timestamp by tone duration (100ms)
            base_timestamp += 0.1
            
            # Add silence between digits (50ms)
            silence = np.zeros(int(0.05 * 8000), dtype=np.float32)
            silence_ulaw = dtmf_to_ulaw(silence)
            await session.send_audio(silence_ulaw, timestamp=base_timestamp)
            
            # Advance timestamp by silence duration (50ms)
            base_timestamp += 0.05
            
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
async def await_call_result(log_id: str, agent:str, idle_timeout_seconds: int = 120, finish_timeout_seconds: int=20,context=None):
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
        finished = False
        while not finished:
            await asyncio.sleep(1)
            log = ChatLog(log_id, agent=agent, user=context.username)

            idle = time.time() - log.last_modified
            logger.debug(f"AWAIT_CALL_RESULT Call session {log_id} idle time: {idle}s")
            if idle >= idle_timeout_seconds:
                logger.info(f"AWAIT_CALL_RESULT Call session {log_id} idle timeout reached ({idle_timeout_seconds}s)")
                finished = True
            commands = log.parsed_commands()
            logger.debug(f"AWAIT_CALL_RESULT Call session {log_id} checking for task_result in commands: {str(commands)}")
            for cmd in commands:
                if 'task_result' in cmd:
                    logger.info(f"AWAIT_CALL_RESULT Call session {log_id} received task_result")
                    log = ChatLog(log_id, agent=agent, user=context.username)
                    log_dump = json.dumps(log.messages)
                    return log_dump
                    #return cmd['task_result']

            user_messages = [msg for msg in log.messages if msg['role'] == 'user']
            logger.debug(f"AWAIT_CALL_RESULT Call session {log_id} checking user messages for CALL DISCONNECTED: {str(user_messages)}")
            for msg in user_messages:
                if msg['content'] and isinstance(msg['content'], list) and len(msg['content']) > 0:
                    text = msg['content'][0].get('text', '')
                    logger.debug(f"AWAIT_CALL_RESULT Call session {log_id} user message content: {text}")
                    if "-- CALL DISCONNECTED --" in text:
                        logger.info(f"AWAIT_CALL_RESULT Call session {log_id} detected CALL DISCONNECTED message")
                        if idle >= finish_timeout_seconds:
                            logger.info(f"AWAIT_CALL_RESULT Call session {log_id} finish timeout reached ({finish_timeout_seconds}s) after disconnect")
                            finished = True

        log = ChatLog(log_id, agent=agent, user=context.username)
        log_dump = json.dumps(log.messages)
        return log_dump
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(f"AWAIT_CALL_RESULT Error in await_call_result: {e}\n\n{trace}")
        return f"Error awaiting call result: {str(e)} \n\n{trace}"

@command()
async def delegate_call_task(agent:str, phone_number:str, instructions: str, idle_timeout_seconds: int = 120,
                             finish_timeout_seconds: int=20, max_call_length_seconds: int = 300,
                             context=None):
    """
    Delegate a task to `agent` to call `phone_number` to accomplish task described in `instructions`.
    Wait for the the call to complete and return the task result from the call
    or the call session log if no task result.

    Example:

    { "delegate_call_task": { 
        "agent": "CustomerService", 
        "phone_number": "16822625850",
        "instructions": "Call the customer and inform them about their order status.",
        "max_call_length_seconds": 300
    }}

    """    
    try:
        log_id = nanoid.generate()
        instructions = instructions + f"\n\n Call the phone number {phone_number} to accomplish the task."
        await command_manager.delegate_task(instructions, agent, log_id=log_id, context=context)
        result = await await_call_result(log_id,agent=agent, idle_timeout_seconds=idle_timeout_seconds, 
                                         finish_timeout_seconds=finish_timeout_seconds, context=context)
        
        # Note: max_call_length handling would need to be added to await_call_result
        # Stop silence monitor before closing S2S session
        try:
            session_manager = get_session_manager()
            session = await session_manager.get_session(log_id)
            if session and session.baresip_bot and hasattr(session.baresip_bot, 'stop_silence_monitor'):
                session.baresip_bot.stop_silence_monitor()
        except Exception as e:
            logger.debug(f"Could not stop silence monitor: {e}")
        
        try:
            await context.close_s2s_session(context)
        except Exception as e:
            logger.warning(f"Could not close s2s session (normal if not s2s): {e}")

        return f"Log_id: {log_id}. Result: {result}"
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(f"Error in delegate_call_task: {e}\n\n{trace}")
        return f"Error delegating call task: {str(e)} \n\n{trace}"

@command()
async def delegate_call_job(agent: str, phone_number: str, instructions: str, 
                            job_type: str = None, timeout: int = 600,
                            idle_timeout_seconds: int = 120, finish_timeout_seconds: int = 20,
                            max_call_length_seconds: int = 300,
                            metadata: dict = None, context=None):
    """
    Delegate a call task to `agent` via the job queue, with call-specific monitoring.
    
    This is like delegate_call_task but uses the job queue for rate limiting and
    concurrency control. It monitors for call completion via:
    - task_result command in the log
    - CALL DISCONNECTED message + finish_timeout
    - idle_timeout
    - job completion/failure
    
    Parameters:
        agent: Name of the agent to handle the call
        phone_number: Phone number to call
        instructions: Task instructions for the agent
        job_type: Optional job type for queue organization (default: "call.{agent}")
        timeout: Maximum seconds to wait for job completion (default: 600)
        idle_timeout_seconds: Seconds of inactivity before considering call done (default: 120)
        finish_timeout_seconds: Seconds to wait after CALL DISCONNECTED (default: 20)
        max_call_length_seconds: Maximum call duration before forced termination (default: 300 = 5 minutes)
        metadata: Optional dict of metadata to attach to the job
    
    Returns:
        Result from the call task or the call session log
    
    Example:
    
    { "delegate_call_job": { 
        "agent": "CustomerService", 
        "phone_number": "16822625850",
        "instructions": "Call the customer and inform them about their order status."
    }}
    """
    try:
        job_id = nanoid.generate()
        
        call_start_time = None  # Will be set when call actually starts
        # Build full instructions with phone number
        full_instructions = instructions + f"\n\n Call the phone number {phone_number} to accomplish the task."
        
        # Default job_type for calls
        if job_type is None:
            job_type = f"call.{agent}"
        
        # Get LLM from context if available
        llm = None
        if context is not None:
            if hasattr(context, 'current_model'):
                llm = context.current_model
            elif hasattr(context, 'data') and 'llm' in context.data:
                llm = context.data['llm']
        
        # Build metadata
        job_metadata = metadata.copy() if metadata else {}
        job_metadata['phone_number'] = phone_number
        job_metadata['call_type'] = 'outbound'
        if context and hasattr(context, 'log_id'):
            job_metadata['parent_log_id'] = context.log_id
        
        # Queue the job
        result = await service_manager.add_job(
            instructions=full_instructions,
            agent_name=agent,
            job_type=job_type,
            username=getattr(context, 'username', None) if context else None,
            metadata=job_metadata,
            job_id=job_id,
            llm=llm,
            context=context
        )
        
        if "error" in result:
            return f"Failed to queue call job: {result['error']}"
        
        queued_job_id = result["job_id"]
        logger.info(f"Queued call job {queued_job_id} for agent {agent} to call {phone_number}")
        
        # Wait for job to start (become active) before monitoring
        # This handles the case where the job queue is backed up
        start_wait = time.time()
        job_started = False
        max_queue_wait = min(timeout, 720)  # Wait up to 12 min for job to start
        while (time.time() - start_wait) < max_queue_wait:
            job_data = await service_manager.get_job_data_service(queued_job_id)
            if job_data:
                status = job_data.get("status")
                if status in ("active", "completed", "failed"):
                    job_started = True
                    logger.info(f"Call job {queued_job_id} is now {status}")
                    break
            await asyncio.sleep(1)
        
        if not job_started:
            logger.warning(f"Call job {queued_job_id} did not start within {timeout}s")
            return f"Job {queued_job_id} did not start within {max_queue_wait}s. It may still be queued."
        
        # Track when call actually started for max_call_length enforcement
        call_start_time = time.time()
        max_call_exceeded = False
        
        # Monitor the call with max_call_length check
        # We'll do our own loop here instead of just calling await_call_result
        # so we can enforce max_call_length
        finished = False
        while not finished:
            await asyncio.sleep(1)
            
            # Check max call length
            call_duration = time.time() - call_start_time
            if call_duration >= max_call_length_seconds:
                logger.info(f"Call job {queued_job_id} exceeded max call length ({max_call_length_seconds}s), terminating")
                max_call_exceeded = True
                
                # Terminate the call
                try:
                    session_manager = get_session_manager()
                    session = await session_manager.get_session(queued_job_id)
                    if session and session.baresip_bot:
                        # Stop silence monitor first
                        if hasattr(session.baresip_bot, 'stop_silence_monitor'):
                            session.baresip_bot.stop_silence_monitor()
                        # Hangup the call
                        await session.baresip_bot.hangup_call()
                        logger.info(f"Call job {queued_job_id} terminated due to max call length")
                except Exception as e:
                    logger.error(f"Error terminating call for max length: {e}")
                
                finished = True
                break
            
            log = ChatLog(queued_job_id, agent=agent, user=context.username)
            idle = time.time() - log.last_modified
            
            if idle >= idle_timeout_seconds:
                logger.info(f"Call job {queued_job_id} idle timeout reached ({idle_timeout_seconds}s)")
                finished = True
                break
                
            commands = log.parsed_commands()
            for cmd in commands:
                if 'task_result' in cmd:
                    logger.info(f"Call job {queued_job_id} received task_result")
                    finished = True
                    break
            
            if finished:
                break
                
            user_messages = [msg for msg in log.messages if msg['role'] == 'user']
            for msg in user_messages:
                if msg['content'] and isinstance(msg['content'], list) and len(msg['content']) > 0:
                    text = msg['content'][0].get('text', '')
                    if "-- CALL DISCONNECTED --" in text:
                        if idle >= finish_timeout_seconds:
                            logger.info(f"Call job {queued_job_id} finish timeout reached after disconnect")
                            finished = True
                            break
        
        # Stop silence monitor before closing S2S session
        try:
            session_manager = get_session_manager()
            session = await session_manager.get_session(queued_job_id)
            if session and session.baresip_bot and hasattr(session.baresip_bot, 'stop_silence_monitor'):
                session.baresip_bot.stop_silence_monitor()
        except Exception as e:
            logger.debug(f"Could not stop silence monitor: {e}")
        
        # Try to close S2S session
        try:
            await context.close_s2s_session(context)
        except Exception as e:
            logger.warning(f"Could not close s2s session (normal if not s2s): {e}")
        
        # Get final log
        log = ChatLog(queued_job_id, agent=agent, user=context.username)
        call_result = json.dumps(log.messages)
        
        # Add max call length exceeded note to result
        if max_call_exceeded:
            actual_duration = time.time() - call_start_time
            exceeded_note = f"\n\n--- CALL TERMINATED: Exceeded maximum call length of {max_call_length_seconds} seconds (actual duration: {actual_duration:.1f}s) ---"
            return f"Job ID: {queued_job_id}. Result: {call_result}{exceeded_note}"
        else:
            return f"Job ID: {queued_job_id}. Result: {call_result}"
        
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(f"Error in delegate_call_job: {e}\n\n{trace}")
        return f"Error delegating call job: {str(e)}\n\n{trace}"
