# Deepgram Integration for mr_sip

## Overview

This document describes the new STT (Speech-to-Text) provider architecture and Deepgram integration for the MindRoot SIP plugin.

## Architecture

### STT Provider Interface

All STT implementations now conform to the `BaseSTTProvider` abstract interface:

```python
class BaseSTTProvider(ABC):
    async def start() -> None
    async def stop() -> None
    async def add_audio(audio_chunk: np.ndarray) -> None
    def set_callbacks(on_partial, on_final) -> None
    def get_stats() -> dict
```

### Available Providers

1. **DeepgramSTT** - Real-time streaming via WebSocket API
   - Latency: 100-300ms for partials, 400-600ms for finals
   - Requires: API key, internet connection
   - Cost: Paid API (pay-as-you-go)

2. **WhisperVADSTT** - Local Whisper with VAD (existing implementation)
   - Latency: 1-3s for finals only
   - Requires: Local compute
   - Cost: Free

3. **StreamingWhisperSTT** - Future: True streaming Whisper
   - Not yet implemented

## File Structure

```
src/mr_sip/
├── stt/
│   ├── __init__.py              # Exports
│   ├── base_stt.py              # Abstract interface
│   ├── stt_factory.py           # Provider factory
│   ├── deepgram_stt.py          # Deepgram implementation ✨ NEW
│   └── whisper_vad_stt.py       # Whisper wrapper ✨ NEW
├── audio/
│   ├── __init__.py
│   ├── inotify_capture.py       # inotify-based capture ✨ NEW
│   └── jack_capture.py          # Future: JACK capture (placeholder)
├── sip_client.py                # Original implementation
├── sip_client_v2.py             # New STT-aware implementation ✨ NEW
├── services.py                  # Original services
└── services_v2.py               # New STT-aware services ✨ NEW
```

## Configuration

### Environment Variables

```bash
# STT Provider Selection
export STT_PROVIDER="deepgram"  # or "whisper_vad"

# Deepgram Configuration (required if using Deepgram)
export DEEPGRAM_API_KEY="your_api_key_here"

# Whisper Configuration (used if STT_PROVIDER="whisper_vad")
export STT_MODEL_SIZE="small"  # tiny, base, small, medium, large

# SIP Configuration (unchanged)
export SIP_GATEWAY="chicago4.voip.ms"
export SIP_USER="498091"
export SIP_PASSWORD="your_password"
```

### Getting a Deepgram API Key

1. Sign up at https://deepgram.com/
2. Navigate to API Keys section
3. Create a new API key
4. Copy and set as `DEEPGRAM_API_KEY` environment variable

### Pricing

Deepgram pricing (as of 2024):
- Nova-2 model: $0.0043/minute
- Pay-as-you-go, no minimum
- First $200 free credits for new accounts

## Usage

### Using Deepgram (Recommended for Low Latency)

```python
# Set environment variables
os.environ['STT_PROVIDER'] = 'deepgram'
os.environ['DEEPGRAM_API_KEY'] = 'your_key_here'

# Make a call (uses dial_service_v2 automatically)
result = await service_manager.dial_service_v2(
    destination="16822625850",
    context=context
)
```

### Using Whisper (Free, Local)

```python
# Set environment variables
os.environ['STT_PROVIDER'] = 'whisper_vad'
os.environ['STT_MODEL_SIZE'] = 'small'

# Make a call
result = await service_manager.dial_service_v2(
    destination="16822625850",
    context=context
)
```

### Backward Compatibility

The original `dial_service` still works and uses the old implementation:

```python
# Old way (still works)
result = await service_manager.dial_service(
    destination="16822625850",
    context=context
)
```

## Audio Capture Methods

### Current: inotify-based File Monitoring

**How it works:**
- baresip writes audio to WAV file
- Linux inotify monitors file for changes
- Audio is read immediately when available
- No polling delay

**Benefits:**
- Much faster than polling (50-100ms improvement)
- Lower CPU usage
- Drop-in replacement for existing system
- Stable and reliable

**Limitations:**
- Still has file I/O overhead
- Requires Linux inotify support

### Future: Direct JACK Capture

**How it would work:**
- Create JACK input port
- Connect to baresip's decoded audio output
- Read audio directly from ring buffer
- Feed to STT provider

**Benefits:**
- Eliminates file I/O completely
- Lowest possible latency
- No disk writes
- Cleaner architecture

**Implementation:**
See `audio/jack_capture.py` for detailed implementation plan.

## Performance Comparison

| Metric | Original | inotify + Whisper | inotify + Deepgram |
|--------|----------|-------------------|--------------------|
| First result | 1-3s | 1-2s | 200-400ms |
| Partial results | No | No | Yes (100-300ms) |
| File I/O | Polling | inotify | inotify |
| CPU usage | Medium | Medium | Low |
| Internet required | No | No | Yes |
| Cost | Free | Free | ~$0.004/min |

## Testing

### Install Dependencies

```bash
# For inotify support
pip install inotify

# For Deepgram
pip install websockets

# Existing dependencies
pip install faster-whisper scipy numpy
```

### Test Deepgram Connection

```python
import asyncio
from mr_sip.stt import create_stt_provider, STTResult
import numpy as np

async def test_deepgram():
    # Create provider
    stt = create_stt_provider('deepgram', api_key='your_key')
    
    # Set callbacks
    def on_partial(result: STTResult):
        print(f"[PARTIAL] {result.text}")
    
    def on_final(result: STTResult):
        print(f"[FINAL] {result.text}")
    
    stt.set_callbacks(on_partial=on_partial, on_final=on_final)
    
    # Start
    await stt.start()
    
    # Send test audio (silence for now)
    for i in range(10):
        audio = np.zeros(1600, dtype=np.float32)  # 100ms of silence
        await stt.add_audio(audio)
        await asyncio.sleep(0.1)
    
    # Stop
    await stt.stop()
    
    # Show stats
    print(stt.get_stats())

asyncio.run(test_deepgram())
```

### Test inotify Capture

```python
import asyncio
from mr_sip.audio import InotifyAudioCapture
import numpy as np

async def test_inotify():
    chunks_received = 0
    
    async def on_chunk(audio: np.ndarray):
        nonlocal chunks_received
        chunks_received += 1
        print(f"Received chunk {chunks_received}: {len(audio)} samples")
    
    # Create capture (assumes audio file exists)
    capture = InotifyAudioCapture(
        audio_file="/path/to/test.wav",
        target_sample_rate=16000,
        chunk_callback=on_chunk
    )
    
    # Start
    await capture.start()
    
    # Wait for some chunks
    await asyncio.sleep(5)
    
    # Stop
    await capture.stop()
    
    print(f"Total chunks: {chunks_received}")
    print(capture.get_stats())

asyncio.run(test_inotify())
```

## Migration Guide

### Step 1: Install Dependencies

```bash
pip install inotify websockets
```

### Step 2: Set Environment Variables

```bash
# For Deepgram
export STT_PROVIDER="deepgram"
export DEEPGRAM_API_KEY="your_key_here"

# OR for Whisper
export STT_PROVIDER="whisper_vad"
export STT_MODEL_SIZE="small"
```

### Step 3: Update Service Calls

Replace `dial_service` with `dial_service_v2` in your code:

```python
# Old
result = await service_manager.dial_service(
    destination=destination,
    context=context
)

# New
result = await service_manager.dial_service_v2(
    destination=destination,
    context=context
)
```

### Step 4: Test

Make a test call and verify:
- Call connects successfully
- Audio is transcribed
- Transcriptions appear in chat
- Latency is acceptable

### Step 5: Monitor

Check logs for:
- STT provider initialization
- Audio capture status
- Transcription results
- Any errors or warnings

## Troubleshooting

### Deepgram Connection Issues

**Problem:** "Failed to connect to Deepgram"

**Solutions:**
- Check API key is correct
- Verify internet connection
- Check firewall allows WebSocket connections
- Try with curl: `curl -H "Authorization: Token YOUR_KEY" https://api.deepgram.com/v1/projects`

### inotify Not Available

**Problem:** "inotify not available"

**Solution:**
```bash
pip install inotify
```

### No Audio Transcribed

**Problem:** Call connects but no transcriptions

**Solutions:**
- Check sndfile module is enabled in baresip config
- Verify audio files are being created (look for dump-*.wav)
- Check STT provider logs for errors
- Verify audio format is correct (16-bit PCM)

### High Latency with Deepgram

**Problem:** Transcriptions are slow

**Solutions:**
- Check internet connection speed
- Try different Deepgram model (nova-2 is fastest)
- Verify audio is being sent continuously
- Check for network congestion

## Future Enhancements

### Phase 1: JACK Audio Capture (Planned)
- Eliminate file I/O completely
- Direct audio capture from baresip
- Lower latency (50-100ms improvement)
- See `audio/jack_capture.py` for plan

### Phase 2: Streaming Whisper (Planned)
- True streaming local transcription
- Partial results with Whisper
- 500-800ms latency
- No internet required

### Phase 3: Additional Providers
- AssemblyAI
- Google Speech-to-Text
- Azure Speech Services
- Custom models

## Support

For issues or questions:
1. Check logs in MindRoot output
2. Review this documentation
3. Check environment variables are set correctly
4. Test with simple audio file first
5. Contact support with logs and configuration
