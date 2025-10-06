#!/usr/bin/env python3
"""
STT (Speech-to-Text) Provider Interface for MindRoot SIP Plugin
"""

from .base_stt import BaseSTTProvider, STTResult
from .stt_factory import create_stt_provider

__all__ = ['BaseSTTProvider', 'STTResult', 'create_stt_provider']
