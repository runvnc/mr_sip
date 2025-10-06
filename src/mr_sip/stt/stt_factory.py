#!/usr/bin/env python3
"""
STT Provider Factory

Creates STT provider instances based on configuration.
"""

import os
import logging
from typing import Optional
from .base_stt import BaseSTTProvider

logger = logging.getLogger(__name__)

def create_stt_provider(provider_name: Optional[str] = None, **kwargs) -> BaseSTTProvider:
    """
    Create an STT provider instance.
    
    Args:
        provider_name: Name of the provider ('deepgram', 'whisper_vad', 'streaming_whisper')
                      If None, uses STT_PROVIDER environment variable or defaults to 'whisper_vad'
        **kwargs: Additional arguments passed to the provider constructor
        
    Returns:
        BaseSTTProvider: Initialized STT provider instance
        
    Environment Variables:
        STT_PROVIDER: Default provider name
        DEEPGRAM_API_KEY: API key for Deepgram (required for deepgram provider)
        STT_MODEL_SIZE: Whisper model size for whisper providers (default: 'small')
    """
    if provider_name is None:
        provider_name = os.getenv('STT_PROVIDER', 'whisper_vad')
        
    provider_name = provider_name.lower()
    
    logger.info(f"Creating STT provider: {provider_name}")
    
    if provider_name == 'deepgram':
        from .deepgram_stt import DeepgramSTT
        api_key = kwargs.get('api_key') or os.getenv('DEEPGRAM_API_KEY')
        if not api_key:
            raise ValueError("Deepgram API key required. Set DEEPGRAM_API_KEY environment variable or pass api_key parameter.")
        return DeepgramSTT(api_key=api_key, **kwargs)
        
    elif provider_name == 'whisper_vad':
        from .whisper_vad_stt import WhisperVADSTT
        model_size = kwargs.get('model_size') or os.getenv('STT_MODEL_SIZE', 'small')
        return WhisperVADSTT(model_size=model_size, **kwargs)
        
    elif provider_name == 'streaming_whisper':
        # Future implementation
        raise NotImplementedError("Streaming Whisper provider not yet implemented. Use 'whisper_vad' or 'deepgram'.")
        
    else:
        raise ValueError(f"Unknown STT provider: {provider_name}. Available: deepgram, whisper_vad, streaming_whisper")
