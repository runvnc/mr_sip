"""
MindRoot SIP Account Wrapper for Incoming Calls

Wraps PySIP's SipAccount to handle incoming calls by creating
MindRoot chat sessions and wiring audio/STT/TTS.
"""
import asyncio
import logging
import os
import traceback
from datetime import datetime
from typing import Optional

import nanoid
from PySIP.sip_account import SipAccount
from PySIP.sip_call import SipCall
from PySIP.filters import CallState

from lib.chatcontext import get_context
from lib.providers.services import service_manager
from .sip_client_v2 import MindRootSIPBotV2
from .sip_manager import get_session_manager

logger = logging.getLogger(__name__)


async def _default_utterance_callback(text: str, utterance_num: int, timestamp: float, ctx, is_eager: bool = False):
    """Default utterance callback for both inbound and outbound calls."""
    try:
        logger.info(f'SIP_DEBUG Transcribed utterance #{utterance_num}: {text}')
        res = await service_manager.cancel_and_wait(ctx.log_id, ctx.username)
        logger.info(f'SIP_DEBUG cancel result: {res}')
        session_manager = get_session_manager()
        session = await session_manager.get_session(ctx.log_id)
        if session:
            session.resume_audio()
        await service_manager.backend_user_message(message=text)
        logger.info(f'SIP_DEBUG Sending message to agent for session {ctx.log_id}')
        await service_manager.send_message_to_agent(session_id=ctx.log_id, message=text, context=ctx)
    except Exception as e:
        logger.error(f'SIP_DEBUG Error processing utterance: {e}')


class MindRootSIPAccount:
    """
    Wraps PySIP SipAccount to handle incoming calls with MindRoot agents.

    Usage:
        account = MindRootSIPAccount(
            user="498091",
            password="secret",
            gateway="chicago4.voip.ms:5060",
            agent_name="Receptionist",
            caller_id="+15551234567"
        )
        await account.start()
        # Now waiting for incoming calls...
        await account.stop()
    """

    def __init__(
        self,
        user: str,
        password: str,
        gateway: str,
        agent_name: str,
        caller_id: str = "",
        stt_provider: str = None,
        stt_config: dict = None,
        enable_recording: bool = False,
        recording_dir: str = 'recordings',
        record_separate: bool = False,
        connection_type: str = "UDP",
        register_duration: int = 600,
    ):
        self.sip_username = user
        self.sip_password = password
        self.sip_server = gateway
        self.agent_name = agent_name
        self.caller_id = caller_id or user
        self.stt_provider_name = stt_provider or os.getenv('STT_PROVIDER', 'deepgram_flux')
        self.stt_config = stt_config or {}
        self.enable_recording = enable_recording
        self.recording_dir = recording_dir
        self.record_separate = record_separate
        self.connection_type = connection_type
        self.register_duration = register_duration

        self._started = False
        self._account: Optional[SipAccount] = None
        self._active_bots: dict = {}

        logger.info(f'MindRootSIPAccount initialized for {user} on {gateway}, agent={agent_name}')

    async def start(self):
        """Register the SIP account and start listening for incoming calls."""
        if self._started:
            logger.warning(f'[INCOMING] start() called on already-started MindRootSIPAccount for {self.sip_username} - ignoring')
            return True

        self._started = True
        logger.info(f'=== STARTING INCOMING CALL LISTENER for {self.sip_username} ===')
        logger.info(f'[INCOMING] Gateway: {self.sip_server}, User: {self.sip_username}, Agent: {self.agent_name}')
        logger.info(f'[INCOMING] Caller ID: {self.caller_id}, Connection: {self.connection_type}')
        logger.info(f'[INCOMING] STT Provider: {self.stt_provider_name}')

        self._account = SipAccount(
            username=self.sip_username,
            password=self.sip_password,
            hostname=self.sip_server,
            caller_id=self.caller_id,
            connection_type=self.connection_type,
            register_duration=self.register_duration,
        )

        # Register the incoming call handler
        self._account.on_incoming_call(self._on_incoming_call)

        # Add a debug message handler to log ALL SIP messages
        def _make_debug_handler():
            async def debug_handler(msg):
                try:
                    method = getattr(msg, 'method', 'N/A')
                    status = getattr(msg, 'status', 'N/A')
                    call_id = getattr(msg, 'call_id', 'N/A')
                    logger.info(f'[INCOMING-DEBUG] SIP msg: method={method}, status={status}, call_id={call_id}')
                except Exception as e:
                    logger.error(f'[INCOMING-DEBUG] Error in debug handler: {e}')
            return debug_handler
        
        if self._account and hasattr(self._account, 'sip_core') and self._account.sip_core:
            self._account.sip_core.on_message_callbacks.append(_make_debug_handler())
            logger.info(f'[INCOMING] Added debug message handler to SipCore')
        else:
            logger.warning(f'[INCOMING] Could not add debug handler - sip_core not available yet')


        # Start the account (registers + starts receive loop)
        logger.info(f'[INCOMING] Calling SipAccount.register()...')
        is_registered = await self._account.register()

        if is_registered:
            logger.info(f'[INCOMING] SIP account REGISTERED successfully. Listening for incoming calls on {self.sip_username}@{self.sip_server}')
            logger.info(f'[INCOMING] Register duration: {self.register_duration}s')
        else:
            logger.error(f'[INCOMING] FAILED to register SIP account {self.sip_username}@{self.sip_server}!')
            raise RuntimeError('SIP registration failed')

        return is_registered

    async def stop(self):
        """Unregister and stop listening for incoming calls."""
        logger.info(f'=== STOPPING INCOMING CALL LISTENER for {self.sip_username} ===')
        logger.info(f'[INCOMING] Active bots to clean up: {len(self._active_bots)}')

        if self._account:
            await self._account.unregister()
            self._account = None

        # Clean up any active bots
        for log_id, bot in list(self._active_bots.items()):
            try:
                await bot.hangup_call()
            except Exception as e:
                logger.warning(f'Error hanging up bot for {log_id}: {e}')
        self._active_bots.clear()

        logger.info('Incoming call listener stopped')

    async def _on_incoming_call(self, call: SipCall):
        """Handle an incoming SIP call.

        This is called by PySIP's SipClient when an INVITE arrives.
        We must call call.accept() (or reject/busy) to resolve the
        call_response_future so PySIP can send the 200 OK.

        IMPORTANT: We accept immediately and do MindRoot setup in a
        background task, because this callback blocks SipClient's
        handle_incoming_call() from proceeding.
        """
        caller_number = getattr(call, 'caller_id', 'unknown') or 'unknown'
        logger.info(f'=== INCOMING CALL from {caller_number} ===')

        try:
            # Accept immediately so PySIP can send 200 OK and start RTP
            logger.info(f'[INCOMING] Calling call.accept() for {caller_number}...')
            await call.accept()
            logger.info(f'[INCOMING] Call from {caller_number} ACCEPTED, starting background setup...')
            logger.info(f'[INCOMING] PySIP should now send 200 OK with SDP')

            # Do MindRoot setup in background so we don't block PySIP
            asyncio.create_task(self._setup_incoming_call(call, caller_number))

        except Exception as e:
            logger.error(f'Error accepting incoming call from {caller_number}: {e}')
            logger.error(traceback.format_exc())
            try:
                await call.reject()
            except Exception:
                pass

    async def _setup_incoming_call(self, call: SipCall, caller_number: str):
        """Set up MindRoot session for an accepted incoming call."""
        logger.info(f'[INCOMING] _setup_incoming_call() started for {caller_number}')
        try:
            # 1. Create a new MindRoot chat session
            log_id = nanoid.generate()
            user = os.getenv('SIP_INCOMING_DEFAULT_USER', 'system')

            logger.info(f'[INCOMING] Creating MindRoot session {log_id} for incoming call from {caller_number}')
            logger.info(f'[INCOMING] User context: {user}, Agent: {self.agent_name}')
            await service_manager.init_chat_session(user, self.agent_name, log_id)
            context = await get_context(log_id, user)

            # 2. Build STT config (same as outbound)
            stt_config = self._build_stt_config()

            # 3. Create the bot
            bot = MindRootSIPBotV2(
                user=self.sip_username,
                password=self.sip_password,
                gateway=self.sip_server,
                on_utterance_callback=_default_utterance_callback,
                stt_provider=self.stt_provider_name,
                stt_config=stt_config,
                context=context,
                enable_recording=self.enable_recording,
                recording_dir=self.recording_dir,
                record_separate=self.record_separate,
            )

            # 4. Attach bot to the call (registers audio callbacks)
            await bot.attach_to_incoming_call(call)
            self._active_bots[log_id] = bot

            # 5. Create SIP session and start audio sender
            session_manager = get_session_manager()
            session = await session_manager.create_session(
                log_id=log_id,
                destination=caller_number,
                baresip_bot=bot
            )
            session.is_active = True
            await session.start_audio_sender()

            logger.info(f'[INCOMING] Session {log_id} FULLY ESTABLISHED for call from {caller_number}')
            logger.info(f'[INCOMING] Bot attached: {bot is not None}')
            logger.info(f'[INCOMING] Audio sender started: True')

        except Exception as e:
            logger.error(f'Error setting up incoming call from {caller_number}: {e}')
            logger.error(traceback.format_exc())
            # Try to hang up if we can't set up properly
            try:
                await call.stop('MindRoot setup failed')
            except Exception:
                pass

    def _build_stt_config(self) -> dict:
        """Build STT configuration (same logic as dial_service_v2)."""
        stt_config = {}
        if self.stt_provider_name in ['deepgram', 'deepgram_flux']:
            stt_config['encoding'] = 'mulaw'
            stt_config['sample_rate'] = 8000
            if os.environ.get('DEEPGRAM_EOT_SECONDS'):
                try:
                    stt_config['eot_threshold'] = float(os.environ.get('DEEPGRAM_EOT_SECONDS'))
                except ValueError:
                    pass
            if os.environ.get('DEEPGRAM_EAGER_EOT_SECONDS'):
                try:
                    stt_config['eager_eot_threshold'] = float(os.environ.get('DEEPGRAM_EAGER_EOT_SECONDS'))
                except ValueError:
                    pass
            stt_config['keyterm'] = ['employee', 'employees', 'employment verification', 'manager', 'HR', 'date-of-birth']
        elif self.stt_provider_name == 'silero_cohere':
            for env_key, cfg_key in [
                ('SILERO_VAD_THRESHOLD', 'threshold'),
                ('SILERO_EAGER_SILENCE_MS', 'eager_silence_ms'),
                ('SILERO_FINAL_SILENCE_MS', 'final_silence_ms'),
                ('SILERO_MIN_SILENCE_MS', 'min_silence_duration_ms'),
                ('SILERO_SPEECH_PAD_MS', 'speech_pad_ms'),
                ('COHERE_TRANSCRIBE_MODEL', 'cohere_model_id'),
                ('COHERE_TRANSCRIBE_LANGUAGE', 'language'),
                ('COHERE_MAX_UTTERANCE_S', 'max_utterance_duration_s'),
                ('COHERE_TRANSCRIBE_URL', 'cohere_transcribe_url'),
            ]:
                val = os.environ.get(env_key)
                if val is not None:
                    stt_config[cfg_key] = val
        return stt_config