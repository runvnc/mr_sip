#!/usr/bin/env python3
"""
Audio Capture and Processing for MindRoot SIP Plugin
"""

from .inotify_capture import InotifyAudioCapture
try:
    # New JACK input capture (does not affect existing TTS JACK output path)
    from .jack_input_capture import JACKAudioCapture
    _HAS_JACK_INPUT = True
except Exception:
    JACKAudioCapture = None  # type: ignore
    _HAS_JACK_INPUT = False

__all__ = ['InotifyAudioCapture'] + ([ 'JACKAudioCapture' ] if _HAS_JACK_INPUT else [])
