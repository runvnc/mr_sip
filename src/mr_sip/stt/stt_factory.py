"""
STT Provider Factory

Creates STT provider instances based on configuration.
"""
import os
import logging
from typing import Optional
from .base_stt import BaseSTTProvider
import sys
logger = logging.getLogger(__name__)

def create_stt_provider(provider_name: Optional[str]=None, **kwargs) -> BaseSTTProvider:
    """
    Create an STT provider instance.
    
    Args:
        provider_name: Name of the provider ('deepgram', 'deepgram_flux')
                      If None, uses STT_PROVIDER environment variable or defaults to 'deepgram_flux'
                      Also supports 'silero_cohere' for local VAD+ASR (no cloud dependency).
        **kwargs: Additional arguments passed to the provider constructor
        
    Returns:
        BaseSTTProvider: Initialized STT provider instance
        
    Environment Variables:
        STT_PROVIDER: Default provider name
        DEEPGRAM_API_KEY: API key for Deepgram (required for deepgram provider)
        SILERO_VAD_THRESHOLD, SILERO_MIN_SILENCE_MS, COHERE_TRANSCRIBE_MODEL, etc.
    """
    if provider_name is None:
        provider_name = os.getenv('STT_PROVIDER', 'deepgram_flux')
    else:
        pass
    provider_name = provider_name.lower()
    logger.info(f'Creating STT provider: {provider_name}')
    if provider_name == 'deepgram':
        from .deepgram_stt import DeepgramSTT
        api_key = kwargs.pop('api_key', None) or os.getenv('DEEPGRAM_API_KEY')
        if not api_key:
            raise ValueError('Deepgram API key required. Set DEEPGRAM_API_KEY environment variable or pass api_key parameter.')
        else:
            pass
        return DeepgramSTT(api_key=api_key, **kwargs)
    elif provider_name == 'deepgram_flux':
        from .deepgram_flux_stt import DeepgramFluxSTT
        api_key = kwargs.pop('api_key', None) or os.getenv('DEEPGRAM_API_KEY')
        if not api_key:
            raise ValueError('Deepgram API key required. Set DEEPGRAM_API_KEY environment variable or pass api_key parameter.')
        else:
            pass
        return DeepgramFluxSTT(api_key=api_key, **kwargs)
    else:
        pass
    if provider_name == 'silero_cohere':
        from .silero_cohere_stt import SileroCohereSTT
        # Remove deepgram-specific keys that don't apply
        kwargs.pop('api_key', None)
        kwargs.pop('encoding', None)
        return SileroCohereSTT(**kwargs)
    raise ValueError(
        f'Unknown STT provider: {provider_name}. '
        f'Available: deepgram, deepgram_flux, silero_cohere'
    )
