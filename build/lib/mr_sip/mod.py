"""
MindRoot SIP Plugin - Main Module

Provides SIP phone integration with MindRoot's AI agent system.
Supports multiple modes:
- Deepgram + separate TTS (v1)
- Deepgram Flux + separate TTS (v2)
- Speech-to-Speech mode (s2s) - for OpenAI Realtime API or similar

V2 now uses PySIP for SIP/RTP handling instead of baresip+JACK.
This eliminates the need for JACK audio server and baresip configuration.

This refactored version imports commands and services from separate modules
for better maintainability and testing.
"""
import logging
import os
MR_DEBUG = os.environ.get('MR_DEBUG', '').lower() in ('1', 'true', 'yes')
if MR_DEBUG:
    logging.getLogger('mr_sip').setLevel(logging.DEBUG)
else:
    logging.getLogger('mr_sip').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
from .commands import *
from .sip_account_wrapper import MindRootSIPAccount
SIP_PROVIDER = os.getenv('SIP_PROVIDER', 'deepgram').lower()
if SIP_PROVIDER == 's2s':
    from .services_s2s import *
elif SIP_PROVIDER == 'deepgram_v2' or os.getenv('SIP_USE_V2', 'true').lower() in ('true', '1', 'yes', 'on'):
    from .services_v2 import *
else:
    from .services_v2 import *
logger.info(f'MindRoot SIP plugin loaded (SIP_PROVIDER={SIP_PROVIDER})')
logger.info(f'[INCOMING] Module loaded. Available commands: start_incoming_calls, stop_incoming_calls, incoming_call_status')
logger.info(f'[INCOMING] Available services: start_incoming_listener_service, stop_incoming_listener_service, get_incoming_listener_status')