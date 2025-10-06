#!/usr/bin/env python3
"""
MindRoot SIP Plugin - Main Module

Provides SIP phone integration with MindRoot's AI agent system.
Enables voice conversations through SIP protocols with real-time transcription and TTS.

This refactored version imports commands and services from separate modules
for better maintainability and testing.
"""

import logging

# Import commands and services for plugin registration
from .commands import *
from .services import *

# Plugin initialization
logger = logging.getLogger(__name__)
logger.info("MindRoot SIP plugin loaded with JACK audio support")
logger.info("Available commands: call, hangup")
logger.info("Available services: dial_service, sip_audio_out_chunk, end_call_service")
