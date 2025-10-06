#!/usr/bin/env python3
"""
Deepgram Flux STT Configuration Example

This file shows how to configure the Deepgram Flux STT provider
for optimal performance in different scenarios.
"""

# Environment variables for Flux STT
FLUX_CONFIG = {
    # Required: Deepgram API key
    'DEEPGRAM_API_KEY': 'your_deepgram_api_key_here',
    
    # STT Provider selection
    'STT_PROVIDER': 'deepgram_flux',  # Use Flux by default
    
    # Optional: Model and language settings
    'FLUX_MODEL': 'flux-general-en',  # Flux model for English
    'FLUX_LANGUAGE': 'en',            # Language code
    
    # Flux-specific tuning parameters
    'FLUX_EAGER_EOT_THRESHOLD': '0.7',  # Eager End-of-Turn threshold (0.3-0.9)
    'FLUX_EOT_THRESHOLD': '0.8',        # Final End-of-Turn threshold
    
    # Audio settings
    'FLUX_SAMPLE_RATE': '16000',        # Audio sample rate
    'FLUX_SMART_FORMAT': 'true',        # Enable smart formatting
    'FLUX_PUNCTUATE': 'true',           # Enable punctuation
}

# Configuration presets for different use cases
CONFIG_PRESETS = {
    # High-speed, low-latency configuration
    'low_latency': {
        'eager_eot_threshold': 0.5,  # Very eager - faster responses, more false starts
        'eot_threshold': 0.7,        # Lower final threshold for speed
        'smart_format': True,
        'punctuate': True,
    },
    
    # Balanced configuration (recommended)
    'balanced': {
        'eager_eot_threshold': 0.7,  # Balanced - good speed with fewer false starts
        'eot_threshold': 0.8,        # Standard final threshold
        'smart_format': True,
        'punctuate': True,
    },
    
    # High-accuracy, conservative configuration
    'high_accuracy': {
        'eager_eot_threshold': 0.8,  # Conservative - fewer false starts, slightly slower
        'eot_threshold': 0.9,        # High final threshold for accuracy
        'smart_format': True,
        'punctuate': True,
    },
    
    # Call center / customer service configuration
    'call_center': {
        'eager_eot_threshold': 0.6,  # Moderate eagerness for natural conversation
        'eot_threshold': 0.8,        # Standard threshold
        'smart_format': True,        # Important for professional transcripts
        'punctuate': True,           # Essential for readability
    },
    
    # Phone assistant / IVR configuration
    'phone_assistant': {
        'eager_eot_threshold': 0.5,  # Very responsive for interactive systems
        'eot_threshold': 0.7,        # Lower threshold for quick responses
        'smart_format': True,
        'punctuate': False,          # May not need punctuation for commands
    }
}

def get_flux_config(preset='balanced'):
    """
    Get Flux configuration for a specific preset.
    
    Args:
        preset: Configuration preset name ('low_latency', 'balanced', 'high_accuracy', 
                'call_center', 'phone_assistant')
                
    Returns:
        dict: Configuration parameters for DeepgramFluxSTT
    """
    if preset not in CONFIG_PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(CONFIG_PRESETS.keys())}")
        
    return CONFIG_PRESETS[preset].copy()

# Usage examples:
"""
# Example 1: Using environment variables
import os
for key, value in FLUX_CONFIG.items():
    os.environ[key] = value

# Example 2: Using preset configuration
from mr_sip.stt import create_stt_provider

# Create STT with balanced preset
config = get_flux_config('balanced')
stt = create_stt_provider('deepgram_flux', **config)

# Example 3: Custom configuration
stt = create_stt_provider(
    'deepgram_flux',
    api_key='your_api_key',
    eager_eot_threshold=0.6,
    eot_threshold=0.8,
    smart_format=True,
    punctuate=True
)
"""

# Performance tuning notes:
"""
TUNING GUIDELINES:

1. eager_eot_threshold (0.3-0.9):
   - Lower values (0.3-0.5): Very responsive, more false starts
   - Medium values (0.6-0.7): Balanced performance (recommended)
   - Higher values (0.8-0.9): Conservative, fewer false starts but slower

2. eot_threshold:
   - Should be higher than eager_eot_threshold
   - Higher values = more confident final decisions
   - Lower values = faster final responses

3. Use Cases:
   - Interactive systems: Lower thresholds for responsiveness
   - Transcription services: Higher thresholds for accuracy
   - Phone calls: Medium thresholds for natural conversation

4. Monitoring:
   - Track EagerEndOfTurn -> TurnResumed ratio
   - High ratio indicates thresholds may be too low
   - Adjust based on your specific use case and user feedback
"""
