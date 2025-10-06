# Deepgram Flux STT Integration

## Overview

The MindRoot SIP plugin now supports Deepgram's new **Flux** model - the first conversational speech recognition model built specifically for voice agents. Flux provides ultra-low latency turn detection with eager end-of-turn processing, making it perfect for phone-based AI agents.

## Key Benefits

- **~260ms end-of-turn detection** (vs traditional STT)
- **Natural conversation flow** with built-in turn-taking
- **Eager End-of-Turn processing** for faster responses
- **Built-in interruption handling** (barge-in support)
- **Nova-3 level accuracy** with conversational optimization

## How Flux Works

Flux uses a sophisticated turn detection system with three key events:

1. **`EagerEndOfTurn`** - Medium confidence the user finished speaking
   - Triggers early LLM processing for faster responses
   - Configurable threshold (0.3-0.9)

2. **`TurnResumed`** - User continued speaking after EagerEndOfTurn
   - Cancels the draft response preparation
   - Waits for next turn event

3. **`EndOfTurn`** - High confidence the user finished speaking
   - Finalizes the transcription and response
   - Guaranteed to match EagerEndOfTurn transcript

## Installation

### 1. Install Dependencies

```bash
cd /xfiles/update_plugins/mr_sip
pip install -r requirements.txt
```

The Deepgram SDK (>=3.0.0) is now included in requirements.txt.

### 2. Set Environment Variables

```bash
# Required: Deepgram API key
export DEEPGRAM_API_KEY="your_deepgram_api_key_here"

# Optional: Set Flux as default STT provider
export STT_PROVIDER="deepgram_flux"
```

### 3. Configuration

Flux is now the **default STT provider** for new installations. You can configure it using:

#### Environment Variables:
```bash
export STT_PROVIDER="deepgram_flux"
export DEEPGRAM_API_KEY="your_api_key"
```

#### Programmatic Configuration:
```python
from mr_sip.stt import create_stt_provider

# Create Flux STT with custom settings
stt = create_stt_provider(
    'deepgram_flux',
    api_key='your_api_key',
    eager_eot_threshold=0.7,  # Eager threshold (0.3-0.9)
    eot_threshold=0.8,        # Final threshold
    smart_format=True,        # Smart formatting
    punctuate=True           # Punctuation
)
```

## Configuration Presets

Use the provided configuration presets for different scenarios:

```python
from config.flux_stt_config import get_flux_config

# Available presets:
config = get_flux_config('balanced')        # Recommended default
config = get_flux_config('low_latency')     # Fastest responses
config = get_flux_config('high_accuracy')   # Most accurate
config = get_flux_config('call_center')     # Customer service
config = get_flux_config('phone_assistant') # Interactive IVR

stt = create_stt_provider('deepgram_flux', **config)
```

## Tuning Parameters

### Core Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `eager_eot_threshold` | 0.3-0.9 | 0.7 | Confidence threshold for EagerEndOfTurn events |
| `eot_threshold` | 0.3-1.0 | 0.8 | Confidence threshold for final EndOfTurn events |
| `model` | string | "flux-general-en" | Flux model variant |
| `language` | string | "en" | Language code |
| `smart_format` | boolean | true | Enable smart formatting |
| `punctuate` | boolean | true | Add punctuation |

### Threshold Tuning Guidelines

#### `eager_eot_threshold` (0.3-0.9):
- **Lower values (0.3-0.5)**: Very responsive, more false starts
- **Medium values (0.6-0.7)**: Balanced performance ✅ **Recommended**
- **Higher values (0.8-0.9)**: Conservative, fewer false starts but slower

#### `eot_threshold`:
- Should be **higher** than `eager_eot_threshold`
- Higher values = more confident final decisions
- Lower values = faster final responses

## Usage Examples

### Basic Usage

```python
import asyncio
from mr_sip.stt import create_stt_provider

async def main():
    # Create Flux STT provider
    stt = create_stt_provider('deepgram_flux')
    
    # Set up event handlers
    def on_partial(result):
        print(f"DRAFT: {result.text}")
        # Start preparing response early
        
    def on_final(result):
        print(f"FINAL: {result.text}")
        # Finalize and send response
        
    stt.on_partial_result = on_partial
    stt.on_final_result = on_final
    
    # Start processing
    await stt.start()
    
    # Send audio chunks (from your audio source)
    # await stt.add_audio(audio_chunk)
    
    await stt.stop()

asyncio.run(main())
```

### Advanced Usage with Custom Configuration

```python
from mr_sip.stt import create_stt_provider
from config.flux_stt_config import get_flux_config

# Use call center preset with custom tweaks
config = get_flux_config('call_center')
config['eager_eot_threshold'] = 0.6  # Slightly more responsive

stt = create_stt_provider('deepgram_flux', **config)

# Enhanced event handling
def handle_eager_eot(result):
    """Handle EagerEndOfTurn - start preparing response."""
    print(f"User likely finished: {result.text}")
    # Start LLM processing speculatively
    start_llm_processing(result.text)
    
def handle_turn_resumed(result):
    """Handle TurnResumed - cancel draft response."""
    print("User continued speaking, canceling draft")
    # Cancel the in-progress LLM call
    cancel_llm_processing()
    
def handle_final(result):
    """Handle EndOfTurn - finalize response."""
    print(f"Final transcript: {result.text}")
    # Deliver the prepared response
    deliver_response()

stt.on_partial_result = handle_eager_eot
stt.on_final_result = handle_final
# Note: TurnResumed events are handled internally
```

## Testing

Run the test script to verify your Flux integration:

```bash
cd /xfiles/update_plugins/mr_sip
export DEEPGRAM_API_KEY="your_api_key"
python test_flux.py
```

## Performance Monitoring

Flux provides detailed statistics for monitoring and tuning:

```python
stats = stt.get_stats()
print(f"""
Flux STT Statistics:
- Provider: {stats['provider']}
- Model: {stats['model']}
- Utterances: {stats['utterance_count']}
- Eager EOTs: {stats['total_eager_eots']}
- Turn Resumed: {stats['total_turn_resumed']}
- Finals: {stats['total_finals']}
- Avg Latency: {stats['average_latency_ms']:.0f}ms
- Eager Threshold: {stats['eager_eot_threshold']}
- EOT Threshold: {stats['eot_threshold']}
""")
```

### Key Metrics to Monitor

1. **EagerEndOfTurn → TurnResumed Ratio**
   - High ratio indicates thresholds may be too low
   - Adjust `eager_eot_threshold` upward if too many false starts

2. **Average Latency**
   - Should be ~260ms for end-of-turn detection
   - Higher latency may indicate network or configuration issues

3. **Response Time Improvement**
   - Compare end-to-end response times with/without eager processing
   - Should see 100-300ms improvement with proper tuning

## Migration from Nova-2

If you're currently using the standard Deepgram provider:

### Before (Nova-2):
```python
stt = create_stt_provider('deepgram', model='nova-2')
```

### After (Flux):
```python
stt = create_stt_provider('deepgram_flux')  # Now the default!
```

### Key Differences:

| Feature | Nova-2 | Flux |
|---------|--------|------|
| **Turn Detection** | Manual (silence-based) | Built-in AI model |
| **Latency** | ~500-1000ms | ~260ms |
| **Partial Results** | Continuous streaming | Eager + Final events |
| **Interruption Handling** | Manual implementation | Built-in barge-in |
| **Conversation Flow** | Basic transcription | Conversational optimization |

## Troubleshooting

### Common Issues

1. **"Failed to connect to Deepgram Flux"**
   - Check your `DEEPGRAM_API_KEY` environment variable
   - Ensure you have Deepgram SDK >=3.0.0 installed
   - Verify your API key has Flux model access

2. **Too many false starts (EagerEndOfTurn → TurnResumed)**
   - Increase `eager_eot_threshold` (try 0.8 or 0.9)
   - Consider using a more conservative preset

3. **Responses too slow**
   - Decrease `eager_eot_threshold` (try 0.5 or 0.6)
   - Use the 'low_latency' preset
   - Check network latency to Deepgram

4. **Import errors**
   - Ensure Deepgram SDK is installed: `pip install deepgram-sdk>=3.0.0`
   - Check Python path includes the mr_sip source directory

### Debug Logging

Enable debug logging to see detailed Flux events:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('mr_sip.stt.deepgram_flux_stt')
logger.setLevel(logging.DEBUG)
```

## Best Practices

1. **Start with 'balanced' preset** and tune from there
2. **Monitor EagerEndOfTurn → TurnResumed ratio** for optimization
3. **Use eager processing judiciously** - it increases LLM calls by ~50-70%
4. **Test with real audio** - silence testing won't trigger turn events
5. **Consider your use case**:
   - Interactive systems: Lower thresholds
   - Transcription services: Higher thresholds
   - Phone calls: Medium thresholds

## Support

For issues specific to Flux integration:
1. Check the logs for detailed error messages
2. Verify your Deepgram API key and model access
3. Test with the provided test script
4. Review the configuration presets for your use case

For Deepgram Flux model questions, refer to:
- [Deepgram Flux Documentation](https://developers.deepgram.com/docs/flux/quickstart)
- [Deepgram Flux Voice Agent Guide](https://developers.deepgram.com/docs/flux/voice-agent-eager-eot)
