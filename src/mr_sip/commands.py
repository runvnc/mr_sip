"""
MindRoot SIP Plugin - User Commands
"""
import os
import logging
from logging.handlers import RotatingFileHandler
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
_s2s_services = None
_v2_services = None

def _get_sip_config(context=None):
    """Get SIP configuration from context or environment."""
    sip_provider = os.getenv('SIP_PROVIDER', 'deepgram').lower()
    require_deepgram = os.getenv('REQUIRE_DEEPGRAM', 'true').lower() in ('true', '1', 'yes', 'on')
    stt_provider = os.getenv('STT_PROVIDER', 'deepgram_flux')
    return (sip_provider, require_deepgram, stt_provider)

def _get_s2s_services():
    """Lazy load S2S services."""
    global _s2s_services
    if _s2s_services is None:
        try:
            from .services_s2s import dial_service as dial_service_s2s, end_call_service as end_call_service_s2s
            _s2s_services = (dial_service_s2s, end_call_service_s2s, True)
        except ImportError:
            _s2s_services = (None, None, False)
        finally:
            pass
    else:
        pass
    return _s2s_services

def _get_v2_services():
    """Lazy load V2 services."""
    global _v2_services
    if _v2_services is None:
        try:
            from .services_v2 import dial_service_v2, end_call_service_v2
            _v2_services = (dial_service_v2, end_call_service_v2, True)
        except Exception as e:
            logger.exception('FATAL: Failed to import V2 SIP services; exiting process')
            raise SystemExit('Fatal mr_sip error: failed to import V2 SIP services; see traceback above') from e
        finally:
            pass
    else:
        pass
    return _v2_services

def _check_s2s_available():
    try:
        from .services_s2s import dial_service as dial_service_s2s, end_call_service as end_call_service_s2s
        return True
    except ImportError:
        return False
    finally:
        pass
logger = logging.getLogger(__name__)
logger.info('Commands module loaded - SIP config will be read from context/environment at runtime')

# ---------------------------------------------------------------------------
# Focused queue/delegate diagnostic channel. Independent of MR_DEBUG so the
# production process can stay at MR_DEBUG=errors without losing lifecycle data.
# Event-level only (never per audio frame) and bounded by rotation.
# ---------------------------------------------------------------------------
DELEGATE_DEBUG_ENABLED = os.getenv(
    'MR_JOB_QUEUE_DIAGNOSTICS', os.getenv('MR_SIP_DELEGATE_DEBUG', '1')
).lower() not in ('0', 'false', 'no', 'off', '')
DELEGATE_DEBUG_LOG = os.getenv(
    'MR_JOB_QUEUE_DIAGNOSTICS_LOG',
    os.getenv('MR_SIP_DELEGATE_DEBUG_LOG', '/tmp/job_queue_diagnostics.log')
)
_delegate_diag_logger = logging.getLogger('mr_sip.delegate_lockup')
_delegate_diag_logger.setLevel(logging.INFO)
_delegate_diag_logger.propagate = False
if DELEGATE_DEBUG_ENABLED and not _delegate_diag_logger.handlers:
    try:
        _delegate_handler = RotatingFileHandler(
            DELEGATE_DEBUG_LOG,
            maxBytes=int(os.getenv('MR_JOB_QUEUE_DIAGNOSTICS_MAX_BYTES', '5242880')),
            backupCount=int(os.getenv('MR_JOB_QUEUE_DIAGNOSTICS_BACKUPS', '3')),
        )
        _delegate_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s', '%Y-%m-%d %H:%M:%S'))
        _delegate_diag_logger.addHandler(_delegate_handler)
    except Exception:
        DELEGATE_DEBUG_ENABLED = False

def _emit_diagnostic(logger, message, *args):
    """Emit to this logger's handlers despite a process-wide logging.disable floor."""
    record = logger.makeRecord(
        logger.name, logging.INFO, __file__, 0, message, args, None
    )
    logger.handle(record)


def _dbg(tag, **fields):
    if not DELEGATE_DEBUG_ENABLED:
        return
    try:
        parts = ' '.join(f'{k}={v!r}' for k, v in fields.items())
        _emit_diagnostic(
            _delegate_diag_logger, 'pid=%s [SIP] %s %s', os.getpid(), tag, parts
        )
    except Exception:
        pass

def _task_snapshot():
    try:
        task = asyncio.current_task()
        coro = task.get_coro() if task else None
        return {
            'done': task.done() if task else None,
            'cancelled': task.cancelled() if task else None,
            'coro': getattr(coro, '__qualname__', repr(coro)),
            'stack': [f'{frame.f_code.co_filename}:{frame.f_lineno}:{frame.f_code.co_name}' for frame in task.get_stack(limit=8)] if task else [],
        }
    except Exception as exc:
        return {'snapshot_error': repr(exc)}

# ---------------------------------------------------------------------------
# Per-number outbound call cooldown (in-memory, resets on server restart).
#
# Unless INTERVAL_VERIFICATION_DAYS is 0 or '0', no call can go out to the same
# number within CALL_COOLDOWN_SECONDS (default 3600 = 1 hour). This prevents
# the AI from rapidly re-dialing a number that returns an instant error.
# The cooldown is recorded at attempt time (before dialing), so failed calls
# also count.
# ---------------------------------------------------------------------------
_recent_call_times = {}  # normalized_number -> time.time() of last attempt
CALL_COOLDOWN_SECONDS = float(os.getenv('MR_SIP_CALL_COOLDOWN_SECONDS', '3600'))


def _cooldown_disabled() -> bool:
    """True if the per-number call cooldown is disabled via env var."""
    val = os.getenv('INTERVAL_VERIFICATION_DAYS', '1')
    return str(val).strip() in ('0', '0.0', '', 'false', 'False', 'no', 'No')


def _check_call_cooldown(destination: str):
    """Return None if OK to call, or an error string if within cooldown."""
    if _cooldown_disabled():
        return None
    now = time.time()
    last = _recent_call_times.get(destination)
    if last is not None:
        elapsed = now - last
        if elapsed < CALL_COOLDOWN_SECONDS:
            logger.warning(
                f'Call to {destination} blocked by cooldown: '
                f'last attempt {int(elapsed)}s ago, cooldown {int(CALL_COOLDOWN_SECONDS)}s'
            )
            return ('NO ANSWER/COULD NOT CONNECT. Note: no retries in this session.')
    return None


def _record_call_attempt(destination: str):
    """Record a call attempt timestamp for cooldown tracking."""
    if _cooldown_disabled():
        return
    _recent_call_times[destination] = time.time()


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
            return 'Error: Destination phone number or SIP URI is required'
        else:
            pass
        if not context or not context.log_id:
            return 'Error: Valid MindRoot context is required for SIP calls'
        else:
            pass
        logger.info(f'Call command initiated to {destination} for session {context.log_id}')
        sip_provider, require_deepgram, stt_provider = _get_sip_config(context)
        logger.info(f'SIP config: provider={sip_provider}, stt={stt_provider}')
        destination = ''.join(filter(str.isalnum, destination + '@'))
        if destination.isdigit() and len(destination) == 10:
            destination = '1' + destination
        else:
            pass
        # Per-number cooldown: prevent rapid re-dialing (e.g. instant-error loop)
        cooldown_error = _check_call_cooldown(destination)
        if cooldown_error:
            return cooldown_error
        else:
            pass
        _record_call_attempt(destination)
        dial_service_s2s, end_call_service_s2s, s2s_available = _get_s2s_services()
        if sip_provider == 's2s' and s2s_available:
            logger.info(f'Using S2S implementation for call to {destination}')
            result = await dial_service_s2s(destination=destination, context=context)
        else:
            dial_service_v2, end_call_service_v2, v2_available = _get_v2_services()
            logger.info(f'Using V2 implementation with STT provider: {stt_provider}')
            if not v2_available or dial_service_v2 is None or not callable(dial_service_v2):
                logger.critical('FATAL: V2 SIP services are not callable; exiting process')
                raise SystemExit('Fatal mr_sip error: V2 SIP services are not callable')
            result = await dial_service_v2(destination=destination, context=context)
        if result['status'] == 'call_established':
            msg = f'Call established to {destination}. Voice conversation is now active. Speak naturally and I will respond through the phone.'
            if result.get('stt_provider'):
                msg += f" (Using {result['stt_provider']} for transcription)"
            else:
                pass
            return None
        elif result['status'] == 'call_failed':
            return f"Failed to establish call to {destination}: {result.get('error', 'Unknown error')}"
        else:
            return f"Call initiation error: {result.get('error', 'Unknown error')}"
    except Exception as e:
        logger.exception(f'Error in call command: {e}')
        return f'Error initiating call: {str(e)}'
    finally:
        pass

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
            return 'Error: Valid MindRoot context is required'
        else:
            pass
        logger.info(f'Hangup command initiated for session {context.log_id}')
        sip_provider, require_deepgram, stt_provider = _get_sip_config(context)
        dial_service_s2s, end_call_service_s2s, s2s_available = _get_s2s_services()
        if sip_provider == 's2s' and s2s_available:
            result = await end_call_service_s2s(context=context)
        else:
            dial_service_v2, end_call_service_v2, v2_available = _get_v2_services()
            result = await end_call_service_v2(context=context)
        if result['status'] == 'call_ended':
            duration = result.get('call_duration_seconds')
            transcript = result.get('transcript', '')
            summary = f'Call ended successfully.'
            if duration:
                summary += f' Duration: {duration:.1f} seconds.'
            else:
                pass
            if transcript:
                summary += f' Transcript captured: {len(transcript.split())} words.'
            else:
                pass
            return summary
        elif result['status'] == 'no_active_call':
            return 'No active call to hang up.'
        else:
            return f"Error ending call: {result.get('error', 'Unknown error')}"
    except Exception as e:
        logger.error(f'Error in hangup command: {e}')
        return f'Error hanging up call: {str(e)}'
    finally:
        pass

def generate_dtmf_tone(digit: str, duration: float=0.1, sample_rate: int=8000) -> np.ndarray:
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
    dtmf_freqs = {'1': (697, 1209), '2': (697, 1336), '3': (697, 1477), '4': (770, 1209), '5': (770, 1336), '6': (770, 1477), '7': (852, 1209), '8': (852, 1336), '9': (852, 1477), '*': (941, 1209), '0': (941, 1336), '#': (941, 1477)}
    if digit not in dtmf_freqs:
        raise ValueError(f'Invalid DTMF digit: {digit}')
    else:
        pass
    low_freq, high_freq = dtmf_freqs[digit]
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    low_tone = np.sin(2 * np.pi * low_freq * t)
    high_tone = np.sin(2 * np.pi * high_freq * t)
    tone = (low_tone + high_tone) / 2.0
    fade_samples = int(0.01 * sample_rate)
    if fade_samples > 0:
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        tone[:fade_samples] *= fade_in
        tone[-fade_samples:] *= fade_out
    else:
        pass
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
    pcm = (tone * 32767).astype(np.int16).tobytes()
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
            logger.error('send_dtmf called without valid context')
            return
        else:
            pass
        if not digits:
            logger.error('send_dtmf called without digits')
            return
        else:
            pass
        valid_dtmf = set('0123456789*#')
        if not all((d in valid_dtmf for d in digits)):
            logger.error(f'Invalid DTMF digits: {digits}')
            return
        else:
            pass
        session_manager = get_session_manager()
        session = await session_manager.get_session(context.log_id)
        if not session or not session.is_active:
            logger.warning(f'No active call for session {context.log_id}')
            return
        else:
            pass
        logger.info(f"Generating DTMF tones for '{digits}'")
        base_timestamp = time_module.perf_counter()
        for digit in digits:
            tone = generate_dtmf_tone(digit, duration=0.1, sample_rate=8000)
            ulaw_data = dtmf_to_ulaw(tone)
            await session.send_audio(ulaw_data, timestamp=base_timestamp)
            base_timestamp += 0.1
            silence = np.zeros(int(0.05 * 8000), dtype=np.float32)
            silence_ulaw = dtmf_to_ulaw(silence)
            await session.send_audio(silence_ulaw, timestamp=base_timestamp)
            base_timestamp += 0.05
            logger.debug(f"Sent DTMF tone for digit '{digit}'")
        else:
            pass
        logger.info(f"Sent DTMF digits '{digits}' for session {context.log_id}")
    except Exception as e:
        logger.error(f'Error in send_dtmf command: {e}')
    finally:
        pass

@command()
async def wait(seconds: float, context=None) -> str:
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
            return 'Error: Valid MindRoot context is required'
        else:
            pass
        if seconds <= 0:
            return 'Error: Wait time must be greater than zero'
        else:
            pass
        logger.info(f'Waiting for {seconds} seconds during SIP call for session {context.log_id}')
        await asyncio.sleep(seconds)
        return f'Waited for {seconds} seconds.'
    except Exception as e:
        logger.error(f'Error in wait command: {e}')
        return f'Error during wait: {str(e)}'
    finally:
        pass


DISCONNECT_MARKER = '-- CALL DISCONNECTED --'


def _message_text(msg) -> str:
    """Best-effort extraction of all text from a chat message regardless of shape.

    Handles content stored as a plain string, a list of parts (dicts with 'text'
    or plain strings), or other/missing shapes. Scans ALL parts, not just [0].
    """
    try:
        content = msg.get('content') if isinstance(msg, dict) else None
    except Exception:
        content = None
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                t = part.get('text')
                if isinstance(t, str):
                    parts.append(t)
        return '\n'.join(parts)
    return str(content)


def _log_has_disconnect(log) -> bool:
    """True if any user message in the log contains the disconnect marker."""
    try:
        for msg in log.messages:
            if msg.get('role') != 'user':
                continue
            if DISCONNECT_MARKER in _message_text(msg):
                return True
    except Exception:
        pass
    return False


# Cleanup awaits (hangup_call / end_session) can hang indefinitely when a call's
# far-end BYE was never cleanly processed (e.g. Bug A mis-routing), leaving a SIP
# teardown awaiting an ACK/transaction that never arrives. A plain try/except does
# NOT protect against a *hang* (only exceptions), so every cleanup await is bounded
# with asyncio.wait_for. This guarantees delegate_call_job / delegate_call_task
# always return instead of stranding the parent job (and its worker slot) forever.
CLEANUP_AWAIT_TIMEOUT = float(os.getenv('MR_SIP_CLEANUP_TIMEOUT', '10'))


async def _await_guarded(coro, what: str, timeout: float = None) -> bool:
    """Await a cleanup coroutine but never block the caller forever.

    Returns True if it completed, False on timeout/error (both logged). On
    timeout the underlying coroutine is cancelled by asyncio.wait_for.
    """
    if timeout is None:
        timeout = CLEANUP_AWAIT_TIMEOUT
    try:
        await asyncio.wait_for(coro, timeout=timeout)
        return True
    except asyncio.TimeoutError:
        _dbg('SIP_CLEANUP_TIMEOUT', what=what, timeout=timeout, task=_task_snapshot())
        logger.error(
            f'CLEANUP TIMEOUT after {timeout}s during: {what}. Continuing; '
            f'SIP teardown may be incomplete (see Bug A / far-end BYE handling).'
        )
        return False
    except Exception as e:
        _dbg('SIP_CLEANUP_ERROR', what=what, error=repr(e), task=_task_snapshot())
        logger.warning(f'Error during {what}: {e}')
        return False


@command()
async def await_call_result(log_id: str, agent: str, idle_timeout_seconds: int=120, finish_timeout_seconds: int=20, context=None):
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
        disconnected_at = None
        while not finished:
            await asyncio.sleep(1)
            log = ChatLog(log_id, agent=agent, user=context.username)
            idle = time.time() - log.last_modified
            logger.debug(f'AWAIT_CALL_RESULT Call session {log_id} idle time: {idle}s')
            if idle >= idle_timeout_seconds:
                logger.info(f'AWAIT_CALL_RESULT Call session {log_id} idle timeout reached ({idle_timeout_seconds}s)')
                finished = True
            else:
                pass
            commands = log.parsed_commands()
            logger.debug(f'AWAIT_CALL_RESULT Call session {log_id} checking for task_result in commands: {str(commands)}')
            for cmd in commands:
                if 'task_result' in cmd or 'hangup' in cmd:
                    logger.info(f'AWAIT_CALL_RESULT Call session {log_id} received {"task_result" if "task_result" in cmd else "hangup"}')
                    log = ChatLog(log_id, agent=agent, user=context.username)
                    log_dump = json.dumps(log.messages)
                    return log_dump
                else:
                    pass
            else:
                pass
            # Latched disconnect detection. Decoupled from log.last_modified because
            # _show_disconnected also pings the agent (send_message_to_agent), which
            # bumps last_modified and would otherwise keep resetting the finish timer.
            if disconnected_at is None and _log_has_disconnect(log):
                disconnected_at = time.time()
                logger.info(f'AWAIT_CALL_RESULT Call session {log_id} detected CALL DISCONNECTED message (latched)')
            if disconnected_at is not None:
                since_disc = time.time() - disconnected_at
                if since_disc >= finish_timeout_seconds:
                    logger.info(f'AWAIT_CALL_RESULT Call session {log_id} finish timeout reached ({finish_timeout_seconds}s); {since_disc:.1f}s since disconnect')
                    finished = True
        log = ChatLog(log_id, agent=agent, user=context.username)
        log_dump = json.dumps(log.messages)
        return log_dump
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(f'AWAIT_CALL_RESULT Error in await_call_result: {e}\n\n{trace}')
        return f'Error awaiting call result: {str(e)} \n\n{trace}'
    finally:
        pass

@command()
async def delegate_call_task(agent: str, phone_number: str, instructions: str, idle_timeout_seconds: int=120, finish_timeout_seconds: int=20, max_call_length_seconds: int=300, context=None):
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
        instructions = instructions + f'\n\n Call the phone number {phone_number} to accomplish the task.'
        await command_manager.delegate_task(instructions, agent, log_id=log_id, context=context)
        result = await await_call_result(log_id, agent=agent, idle_timeout_seconds=idle_timeout_seconds, finish_timeout_seconds=finish_timeout_seconds, context=context)
        try:
            session_manager = get_session_manager()
            session = await session_manager.get_session(log_id)
            if session and session.baresip_bot:
                if hasattr(session.baresip_bot, 'stop_silence_monitor'):
                    session.baresip_bot.stop_silence_monitor()
                # Always hang up the call on exit so the recorder (write-at-end)
                # is flushed to disk and the SIP session is torn down cleanly,
                # regardless of why the wait loop ended (idle timeout, task_result, etc.).
                await _await_guarded(session.baresip_bot.hangup_call(),
                                     f'hangup_call cleanup ({log_id})')
                await _await_guarded(session_manager.end_session(log_id),
                                     f'end_session cleanup ({log_id})')
            else:
                pass
        except Exception as e:
            logger.debug(f'Could not stop silence monitor / hang up call: {e}')
        finally:
            pass
        try:
            await context.close_s2s_session(context)
        except Exception as e:
            logger.warning(f'Could not close s2s session (normal if not s2s): {e}')
        finally:
            pass
        return f'Log_id: {log_id}. Result: {result}'
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(f'Error in delegate_call_task: {e}\n\n{trace}')
        return f'Error delegating call task: {str(e)} \n\n{trace}'
    finally:
        pass

@command()
async def delegate_call_job(agent: str, phone_number: str, instructions: str, job_type: str=None, timeout: int=600, idle_timeout_seconds: int=120, finish_timeout_seconds: int=20, max_call_length_seconds: int=300, metadata: dict=None, context=None):
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
        call_start_time = None
        _dbg('DCJ_ENTER', job_id=job_id, agent=agent, phone=phone_number, job_type=job_type,
             timeout=timeout, idle_timeout=idle_timeout_seconds,
             finish_timeout=finish_timeout_seconds, max_call_length=max_call_length_seconds,
             parent_log=getattr(context, 'log_id', None))
        full_instructions = instructions + f'\n\n Call the phone number {phone_number} to accomplish the task.'
        if job_type is None:
            job_type = f'call.{agent}'
        else:
            pass
        llm = None
        if context is not None:
            if hasattr(context, 'current_model'):
                llm = context.current_model
            elif hasattr(context, 'data') and 'llm' in context.data:
                llm = context.data['llm']
            else:
                pass
        else:
            pass
        job_metadata = metadata.copy() if metadata else {}
        job_metadata['phone_number'] = phone_number
        job_metadata['call_type'] = 'outbound'
        if context and hasattr(context, 'log_id'):
            job_metadata['parent_log_id'] = context.log_id
        else:
            pass
        result = await service_manager.add_job(instructions=full_instructions, agent_name=agent, job_type=job_type, username=getattr(context, 'username', None) if context else None, metadata=job_metadata, job_id=job_id, llm=llm, context=context)
        if 'error' in result:
            return f"Failed to queue call job: {result['error']}"
        else:
            pass
        queued_job_id = result['job_id']
        logger.info(f'Queued call job {queued_job_id} for agent {agent} to call {phone_number}')
        start_wait = time.time()
        job_started = False
        # Bind before the loop: if timeout<=0 the loop body never runs and the
        # DCJ_DID_NOT_START trace (last_status=status) would raise NameError.
        status = None
        max_queue_wait = min(timeout, 420)  # 7 minutes max
        while time.time() - start_wait < max_queue_wait:
            job_data = await service_manager.get_job_data_service(queued_job_id)
            status = job_data.get('status') if job_data else None
            elapsed = time.time() - start_wait
            if int(elapsed) % 5 == 0:
                _dbg('DCJ_QUEUE_POLL', job_id=queued_job_id, elapsed=round(elapsed, 1),
                     status=status, found=bool(job_data))
            if job_data:
                if status in ('active', 'completed', 'failed'):
                    job_started = True
                    _dbg('DCJ_JOB_STARTED', job_id=queued_job_id, status=status,
                         elapsed=round(elapsed, 1))
                    logger.info(f'Call job {queued_job_id} is now {status}')
                    break
                else:
                    pass
            else:
                pass
            await asyncio.sleep(1)
        else:
            pass
        if not job_started:
            _dbg('DCJ_DID_NOT_START', job_id=queued_job_id, waited=max_queue_wait, last_status=status)
            logger.warning(f'Call job {queued_job_id} did not start within {max_queue_wait}s (waited full queue window)')
            # Cancel the orphaned queued job
            cancel_succeeded = False
            try:
                from lib.providers.commands import command_manager
                await command_manager.cancel_job(queued_job_id)
                logger.info(f'Cancelled orphaned call job {queued_job_id}')
                cancel_succeeded = True
            except Exception as ce:
                logger.warning(f'Could not cancel orphaned job {queued_job_id}: {ce}')
            if cancel_succeeded:
                _dbg('DCJ_DID_NOT_START_RETURN', job_id=queued_job_id, cancelled=True)
                return (
                    f'CALL FAILED: the call worker did not pick up the job within {max_queue_wait}s '
                    f'(job_id: {queued_job_id}); the queued job was cancelled. Record this as a FAILED '
                    f'call attempt (no connection was made) and move on. DO NOT retry or redial this number.'
                )
            else:
                _dbg('DCJ_DID_NOT_START_RETURN', job_id=queued_job_id, cancelled=False)
                return (
                    f'CALL FAILED: the call worker did not pick up the job within {max_queue_wait}s and it '
                    f'could NOT be cancelled (job_id: {queued_job_id}). Record this as a FAILED call attempt '
                    f'and move on. DO NOT retry or redial this number (a duplicate call may already be in progress).'
                )
        else:
            pass
        call_start_time = time.time()
        max_call_exceeded = False
        finished = False
        disconnected_at = None
        _dbg('DCJ_MONITOR_START', job_id=queued_job_id, max_call_length=max_call_length_seconds,
             idle_timeout=idle_timeout_seconds, finish_timeout=finish_timeout_seconds)
        _last_mon_log = -999.0
        while not finished:
            await asyncio.sleep(1)
            call_duration = time.time() - call_start_time
            if call_duration - _last_mon_log >= 5:
                _last_mon_log = call_duration
                try:
                    _mlog = ChatLog(queued_job_id, agent=agent, user=context.username)
                    _idle = time.time() - _mlog.last_modified
                except Exception:
                    _idle = None
                _dbg('DCJ_MONITOR_TICK', job_id=queued_job_id, duration=round(call_duration, 1),
                     idle=(round(_idle, 1) if _idle is not None else None),
                     since_disc=(round(time.time() - disconnected_at, 1) if disconnected_at else None))
            if call_duration >= max_call_length_seconds:
                logger.info(f'Call job {queued_job_id} exceeded max call length ({max_call_length_seconds}s), terminating')
                max_call_exceeded = True
                try:
                    session_manager = get_session_manager()
                    session = await session_manager.get_session(queued_job_id)
                    if session and session.baresip_bot:
                        if hasattr(session.baresip_bot, 'stop_silence_monitor'):
                            session.baresip_bot.stop_silence_monitor()
                        else:
                            pass
                        await _await_guarded(session.baresip_bot.hangup_call(),
                                             f'max-length hangup_call ({queued_job_id})')
                        logger.info(f'Call job {queued_job_id} terminated due to max call length')
                    else:
                        pass
                except Exception as e:
                    logger.error(f'Error terminating call for max length: {e}')
                finally:
                    pass
                finished = True
                break
            else:
                pass
            log = ChatLog(queued_job_id, agent=agent, user=context.username)
            idle = time.time() - log.last_modified
            if idle >= idle_timeout_seconds:
                logger.info(f'Call job {queued_job_id} idle timeout reached ({idle_timeout_seconds}s)')
                finished = True
                break
            else:
                pass
            commands = log.parsed_commands()
            for cmd in commands:
                if 'task_result' in cmd or 'hangup' in cmd:
                    logger.info(f'Call job {queued_job_id} received {"task_result" if "task_result" in cmd else "hangup"}')
                    finished = True
                    break
                else:
                    pass
            else:
                pass
            if finished:
                break
            else:
                pass
            # Latched disconnect detection (decoupled from last_modified; see await_call_result).
            if disconnected_at is None and _log_has_disconnect(log):
                disconnected_at = time.time()
                logger.info(f'Call job {queued_job_id} detected CALL DISCONNECTED message (latched)')
            if disconnected_at is not None:
                since_disc = time.time() - disconnected_at
                if since_disc >= finish_timeout_seconds:
                    logger.info(f'Call job {queued_job_id} finish timeout reached ({finish_timeout_seconds}s); {since_disc:.1f}s since disconnect')
                    finished = True
                    break
        _dbg('DCJ_MONITOR_EXIT', job_id=queued_job_id,
             duration=round(time.time() - call_start_time, 1),
             max_call_exceeded=max_call_exceeded,
             disconnected=(disconnected_at is not None))
        _dbg('DCJ_CLEANUP_GET_SESSION_BEGIN', job_id=queued_job_id, task=_task_snapshot())
        try:
            session_manager = get_session_manager()
            session = await asyncio.wait_for(session_manager.get_session(queued_job_id), timeout=CLEANUP_AWAIT_TIMEOUT)
            _dbg('DCJ_CLEANUP_GET_SESSION_DONE', job_id=queued_job_id, has_session=bool(session))
            if session and session.baresip_bot:
                if hasattr(session.baresip_bot, 'stop_silence_monitor'):
                    session.baresip_bot.stop_silence_monitor()
                # Always hang up the call on exit so the recorder (write-at-end)
                # is flushed to disk and the SIP session is torn down cleanly,
                # regardless of why the wait loop ended (idle timeout, task_result, etc.).
                await _await_guarded(session.baresip_bot.hangup_call(),
                                     f'hangup_call cleanup ({queued_job_id})')
                await _await_guarded(session_manager.end_session(queued_job_id),
                                     f'end_session cleanup ({queued_job_id})')
            else:
                pass
        except Exception as e:
            logger.debug(f'Could not stop silence monitor / hang up call: {e}')
        finally:
            pass
        try:
            await context.close_s2s_session(context)
        except Exception as e:
            logger.warning(f'Could not close s2s session (normal if not s2s): {e}')
        finally:
            pass
        log = ChatLog(queued_job_id, agent=agent, user=context.username)
        call_result = json.dumps(log.messages)
        _dbg('DCJ_RETURN', job_id=queued_job_id, max_call_exceeded=max_call_exceeded,
             result_len=len(call_result))
        if max_call_exceeded:
            actual_duration = time.time() - call_start_time
            exceeded_note = f'\n\n--- CALL TERMINATED: Exceeded maximum call length of {max_call_length_seconds} seconds (actual duration: {actual_duration:.1f}s) ---'
            return f'Job ID: {queued_job_id}. Result: {call_result}{exceeded_note}'
        else:
            return f'Job ID: {queued_job_id}. Result: {call_result}'
    except asyncio.CancelledError:
        _dbg('DCJ_CANCELLED', job_id=locals().get('queued_job_id', None),
             note='coroutine cancelled (SSE/turn cancel or process shutdown) - NOT returning to caller')
        raise
    except Exception as e:
        trace = traceback.format_exc()
        _dbg('DCJ_EXCEPTION', job_id=locals().get('queued_job_id', None), error=str(e))
        logger.error(f'Error in delegate_call_job: {e}\n\n{trace}')
        return f'Error delegating call job: {str(e)}\n\n{trace}'
    finally:
        _dbg('DCJ_FINALLY', job_id=locals().get('queued_job_id', None))


@command()
async def start_incoming_calls(agent: str = None, context=None) -> str:
    """
    Start listening for incoming SIP calls.
    
    DEBUG: This command was invoked.
    
    Registers the SIP account with the provider and sets up a listener
    that creates MindRoot chat sessions for each incoming call.
    
    Args:
        agent: Which MindRoot agent answers incoming calls.
               Defaults to SIP_INCOMING_AGENT env var.
        context: MindRoot context (automatically provided)
    
    Returns:
        str: Status message
    
    Example:
        { "start_incoming_calls": { "agent": "Receptionist" } }
    
    Environment Variables:
        SIP_INCOMING_AGENT: Default agent for incoming calls
        SIP_GATEWAY: SIP gateway server
        SIP_USER: SIP username
        SIP_PASSWORD: SIP password
    """
    try:
        logger.info(f'[INCOMING-CMD] start_incoming_calls invoked with agent={agent}')
        result = await service_manager.start_incoming_listener_service(agent_name=agent, context=context)
        if result['status'] == 'started':
            logger.info(f'[INCOMING-CMD] Listener started: {result}')
            return f"Incoming call listener started. Agent: {result['agent']}, Gateway: {result['gateway']}, Caller ID: {result['caller_id']}"
        elif result['status'] == 'already_running':
            return "Incoming call listener is already running."
        else:
            return f"Failed to start incoming call listener: {result.get('error', 'Unknown error')}"
    except Exception as e:
        logger.error(f'Error in start_incoming_calls: {e}')
        return f'Error starting incoming call listener: {str(e)}'

@command()
async def stop_incoming_calls(context=None) -> str:
    """
    Stop listening for incoming SIP calls.
    
    DEBUG: This command was invoked.
    
    Unregisters from the SIP provider and stops the incoming call listener.
    Active calls are NOT terminated.
    
    Args:
        context: MindRoot context (automatically provided)
    
    Returns:
        str: Status message
    
    Example:
        { "stop_incoming_calls": {} }
    """
    try:
        result = await service_manager.stop_incoming_listener_service(context=context)
        if result['status'] == 'stopped':
            return "Incoming call listener stopped."
        elif result['status'] == 'not_running':
            return "Incoming call listener is not running."
        else:
            return f"Failed to stop incoming call listener: {result.get('error', 'Unknown error')}"
    except Exception as e:
        logger.error(f'Error in stop_incoming_calls: {e}')
        return f'Error stopping incoming call listener: {str(e)}'

@command()
async def incoming_call_status(context=None) -> str:
    """
    Get the status of the incoming call listener.
    
    Args:
        context: MindRoot context (automatically provided)
    
    Returns:
        str: Status message
    
    Example:
        { "incoming_call_status": {} }
    """
    try:
        result = await service_manager.get_incoming_listener_status(context=context)
        if result['is_active']:
            return f"Incoming call listener is ACTIVE. Agent: {result['agent']}, Gateway: {result['gateway']}, User: {result['user']}"
        else:
            return "Incoming call listener is NOT running."
    except Exception as e:
        logger.error(f'Error in incoming_call_status: {e}')
        return f'Error getting incoming call status: {str(e)}'


def _load_wav_as_ulaw_8k(file_path: str, channel: str = 'left') -> bytes:
    """Load a WAV file and return PCMU (ulaw) 8kHz mono bytes.

    Handles arbitrary sample width / channel count / sample rate via audioop.
    For stereo input, `channel` selects 'left', 'right', or 'mix'. In the
    project's dual-channel call recordings the LEFT channel is the far-end
    (caller), so 'left' is the default.
    """
    import wave
    import audioop

    with wave.open(file_path, 'rb') as wf:
        n_channels = wf.getnchannels()
        samp_width = wf.getsampwidth()
        frame_rate = wf.getframerate()
        pcm = wf.readframes(wf.getnframes())

    # Normalize to 16-bit PCM.
    if samp_width != 2:
        pcm = audioop.lin2lin(pcm, samp_width, 2)
        samp_width = 2

    # Downmix / select channel to mono.
    if n_channels == 2:
        ch = (channel or 'left').lower()
        if ch == 'left':
            pcm = audioop.tomono(pcm, 2, 1.0, 0.0)
        elif ch == 'right':
            pcm = audioop.tomono(pcm, 2, 0.0, 1.0)
        else:  # mix
            pcm = audioop.tomono(pcm, 2, 0.5, 0.5)
    elif n_channels != 1:
        # >2 channels: fold everything down naively to mono.
        pcm = audioop.tomono(pcm, 2, 0.5, 0.5)

    # Resample to 8kHz.
    if frame_rate != 8000:
        pcm, _ = audioop.ratecv(pcm, 2, 1, frame_rate, 8000, None)

    return audioop.lin2ulaw(pcm, 2)


@command()
async def play_audio(file_path: str, channel: str = 'mix', wait: bool = True,
                     bargeable: bool = True, context=None) -> str:
    """
    Play a WAV file OUT to the caller on the active call (outbound audio).

    The clip is converted to PCMU
    (ulaw) 8kHz mono and streamed, paced at real time in 20ms frames, out the
    SAME outbound path the TTS uses (bot.start_tts_response -> send_tts_audio ->
    end_tts_response -> PySIP RTP). The far end hears it. Useful for an incoming
    receptionist agent to play a scripted clip to the caller (e.g. to test the
    other side's barge-in / cross-talk VAD with controlled, repeatable audio).

    Args:
        file_path: Absolute path to a WAV file (any rate/width/channels).
        channel:   For stereo input, which channel to play: 'mix' (default),
                   'left', or 'right'.
        wait:      If True (default), block until the whole clip has played, so
                   the agent's next action happens after playback. If False,
                   play in the background and return immediately.
        bargeable: If True (default), a barge-in (far-end speech) halts playback
                   like normal TTS. If False, the clip plays straight through and
                   far-end speech does NOT interrupt it - use this for the
                   receptionist test rig so a scripted clip faithfully simulates
                   a human who keeps talking over the AI (the default rig behavior
                   of stopping on any caller audio cannot reproduce that).
        context:   MindRoot context (automatically provided).

    Returns:
        str: Status message with the converted duration and frame count.

    Example:
        { "play_audio": { "file_path": "/path/greeting.wav" } }
        { "play_audio": { "file_path": "/path/clip.wav", "wait": false } }
    """
    try:
        if not context or not context.log_id:
            return 'Error: Valid MindRoot context is required'
        if not file_path or not os.path.exists(file_path):
            return f'Error: WAV file not found: {file_path}'

        session_manager = get_session_manager()
        session = await session_manager.get_session(context.log_id)
        if not session or not session.is_active:
            return f'Error: No active call for session {context.log_id}'

        bot = session.baresip_bot
        if not bot:
            return 'Error: No SIP bot attached to this session'
        for _m in ('start_tts_response', 'send_tts_audio', 'end_tts_response'):
            if not hasattr(bot, _m):
                return f'Error: bot does not support {_m} (outbound playback unavailable on this bot type)'

        try:
            ulaw = _load_wav_as_ulaw_8k(file_path, channel=channel)
        except Exception as e:
            logger.error(f'play_audio: failed to load/convert {file_path}: {e}')
            return f'Error loading WAV: {e}'

        frame_bytes = 160  # 20ms at ulaw 8kHz
        n_frames = (len(ulaw) + frame_bytes - 1) // frame_bytes
        duration_s = len(ulaw) / 8000.0

        logger.info(
            f'play_audio: playing {file_path} ({duration_s:.2f}s, {n_frames} '
            f'x 20ms frames) OUT to caller for session {context.log_id} '
            f'(channel={channel}, wait={wait})'
        )

        async def _play():
            frame_dt = 0.02
            sent = 0
            if not bargeable:
                try:
                    setattr(bot, '_playback_locked', True)
                    logger.info('play_audio: bargeable=False -> playback locked (barge-in disabled)')
                except Exception:
                    pass
            try:
                await bot.start_tts_response()
                start = time.perf_counter()
                for i in range(0, len(ulaw), frame_bytes):
                    if not session.is_active:
                        logger.info('play_audio: call no longer active, stopping playback')
                        break
                    frame = ulaw[i:i + frame_bytes]
                    ts = start + sent * frame_dt
                    await bot.send_tts_audio(frame, timestamp=ts)
                    sent += 1
                    next_t = start + sent * frame_dt
                    delay = next_t - time.perf_counter()
                    if delay > 0:
                        await asyncio.sleep(delay)
            except Exception as e:
                logger.error(f'play_audio: error during playback: {e}\n{traceback.format_exc()}')
            finally:
                try:
                    await bot.end_tts_response()
                except Exception:
                    pass
                if not bargeable:
                    try:
                        setattr(bot, '_playback_locked', False)
                        logger.info('play_audio: playback unlocked (barge-in re-enabled)')
                    except Exception:
                        pass
            logger.info(f'play_audio: done, played {sent}/{n_frames} frames for session {context.log_id}')

        if wait:
            await _play()
            return (f'Played {file_path} ({duration_s:.2f}s, {n_frames} frames) '
                    f'out to caller for session {context.log_id}.')
        else:
            asyncio.create_task(_play())
            return (f'Started playing {file_path} ({duration_s:.2f}s, {n_frames} '
                    f'frames) out to caller for session {context.log_id} '
                    f'(background, realtime pace).')
    except Exception as e:
        logger.error(f'Error in play_audio: {e}\n{traceback.format_exc()}')
        return f'Error playing audio: {str(e)}'
