# MindRoot SIP Plugin V2 - Deepgram Integration

## What's New in V2

### 🚀 Key Features

1. **Abstract STT Provider Interface**
   - Easy switching between STT backends
   - Deepgram, Whisper, and future providers
   - Consistent API across all providers

2. **Deepgram Streaming STT**
   - Real-time transcription with 100-300ms latency
   - Partial results for immediate feedback
   - High accuracy with cloud-based processing

3. **inotify-based Audio Capture**
   - Eliminates polling delays
   - Lower CPU usage
   - Immediate notification of new audio data

4. **Backward Compatibility**
   - Original implementation still available
   - Gradual migration path
   - No breaking changes to existing code

## Quick Start

### 1. Install Dependencies

```bash
cd /xfiles/update_plugins/mr_sip
pip install -r requirements.txt
```

### 2. Configure STT Provider

#### Option A: Deepgram (Recommended for Low Latency)

```bash
export STT_PROVIDER="deepgram"
export DEEPGRAM_API_KEY="your_deepgram_api_key"
```

Get your API key from: https://deepgram.com/

#### Option B: Whisper (Free, Local)

```bash
export STT_PROVIDER="whisper_vad"
export STT_MODEL_SIZE="small"  # tiny, base, small, medium, large
```

### 3. Make a Call

```python
from lib.providers.services import service_manager

# Using V2 with STT provider interface
result = await service_manager.dial_service_v2(
    destination="16822625850",
    context=context
)

# Or use original (still works)
result = await service_manager.dial_service(
    destination="16822625850",
    context=context
)
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     MindRoot Agent                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  SIP Client V2                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              STT Provider Interface                   │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐     │  │
│  │  │  Deepgram  │  │  Whisper   │  │   Future   │     │  │
│  │  │    STT     │  │  VAD STT   │  │  Providers │     │  │
│  │  └────────────┘  └────────────┘  └────────────┘     │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Audio Capture (inotify)                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  baresip → WAV file → inotify → Audio Buffer         │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## File Structure

```
mr_sip/
├── stt/                          # STT Provider Interface
│   ├── __init__.py
│   ├── base_stt.py              # Abstract base class
│   ├── stt_factory.py           # Provider factory
│   ├── deepgram_stt.py          # Deepgram implementation ✨
│   └── whisper_vad_stt.py       # Whisper wrapper ✨
│
├── audio/                        # Audio Capture
│   ├── __init__.py
│   ├── inotify_capture.py       # inotify-based capture ✨
│   └── jack_capture.py          # Future: JACK capture
│
├── sip_client.py                # Original implementation
├── sip_client_v2.py             # New STT-aware client ✨
├── services.py                  # Original services
├── services_v2.py               # New STT-aware services ✨
│
└── [other existing files...]
```

## Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `STT_PROVIDER` | STT provider to use | `whisper_vad` | No |
| `DEEPGRAM_API_KEY` | Deepgram API key | - | Yes (if using Deepgram) |
| `STT_MODEL_SIZE` | Whisper model size | `small` | No |
| `SIP_GATEWAY` | SIP gateway server | `chicago4.voip.ms` | No |
| `SIP_USER` | SIP username | `498091` | No |
| `SIP_PASSWORD` | SIP password | - | Yes |
| `AUDIO_DIR` | Audio recording directory | `.` | No |

### STT Provider Options

#### Deepgram

```bash
export STT_PROVIDER="deepgram"
export DEEPGRAM_API_KEY="your_key"
```

**Pros:**
- Lowest latency (100-300ms)
- Partial results
- High accuracy
- No local compute needed

**Cons:**
- Requires internet
- Costs money (~$0.004/min)
- Requires API key

#### Whisper VAD

```bash
export STT_PROVIDER="whisper_vad"
export STT_MODEL_SIZE="small"
```

**Pros:**
- Free
- Works offline
- Private (no data sent to cloud)
- Good accuracy

**Cons:**
- Higher latency (1-3s)
- No partial results
- Requires local compute

## Performance Comparison

| Metric | Original | V2 + Whisper | V2 + Deepgram |
|--------|----------|--------------|---------------|
| First result | 1-3s | 1-2s | 200-400ms |
| Partial results | ❌ | ❌ | ✅ (100-300ms) |
| Audio capture | Polling | inotify | inotify |
| Latency improvement | - | ~30% | ~75% |
| Internet required | ❌ | ❌ | ✅ |
| Cost | Free | Free | ~$0.004/min |

## Usage Examples

### Basic Call with Deepgram

```python
import os
import asyncio
from lib.providers.services import service_manager

# Configure
os.environ['STT_PROVIDER'] = 'deepgram'
os.environ['DEEPGRAM_API_KEY'] = 'your_key_here'

async def make_call():
    result = await service_manager.dial_service_v2(
        destination="16822625850",
        context=context
    )
    
    if result['status'] == 'call_established':
        print(f"Call connected! Using {result['stt_provider']}")
    else:
        print(f"Call failed: {result.get('error')}")

asyncio.run(make_call())
```

### Custom STT Configuration

```python
from mr_sip.stt import create_stt_provider, STTResult

# Create custom Deepgram provider
stt = create_stt_provider(
    'deepgram',
    api_key='your_key',
    model='nova-2',
    language='en-US',
    interim_results=True
)

# Set callbacks
def on_partial(result: STTResult):
    print(f"[PARTIAL] {result.text}")

def on_final(result: STTResult):
    print(f"[FINAL] {result.text} (confidence: {result.confidence})")

stt.set_callbacks(on_partial=on_partial, on_final=on_final)

# Use in call
await stt.start()
# ... feed audio ...
await stt.stop()
```

### Testing STT Provider

```python
import numpy as np
from mr_sip.stt import create_stt_provider

async def test_stt():
    # Create provider
    stt = create_stt_provider('deepgram', api_key='your_key')
    
    # Start
    await stt.start()
    
    # Send test audio (silence)
    for i in range(10):
        audio = np.zeros(1600, dtype=np.float32)  # 100ms
        await stt.add_audio(audio)
        await asyncio.sleep(0.1)
    
    # Stop and show stats
    await stt.stop()
    print(stt.get_stats())

asyncio.run(test_stt())
```

## Migration from V1

See [MIGRATION_CHECKLIST.md](MIGRATION_CHECKLIST.md) for detailed steps.

### Quick Migration

1. Install dependencies: `pip install inotify websockets`
2. Set environment variables (see Configuration)
3. Replace `dial_service` with `dial_service_v2`
4. Test thoroughly

### Rollback

If issues occur, simply:
1. Revert to `dial_service` (original)
2. Remove new environment variables
3. Restart services

The original implementation is unchanged and still available.

## Troubleshooting

### Deepgram Connection Failed

**Symptoms:** "Failed to connect to Deepgram"

**Solutions:**
- Verify API key: `echo $DEEPGRAM_API_KEY`
- Test API key:
  ```bash
  curl -H "Authorization: Token $DEEPGRAM_API_KEY" \
       https://api.deepgram.com/v1/projects
  ```
- Check internet connection
- Verify firewall allows WebSocket connections

### No Transcriptions

**Symptoms:** Call connects but no text appears

**Solutions:**
- Check STT provider is set: `echo $STT_PROVIDER`
- Verify audio files are created: `ls -la dump-*.wav`
- Check logs for STT errors
- Ensure sndfile module is enabled in baresip config

### inotify Not Available

**Symptoms:** "inotify not available" error

**Solution:**
```bash
pip install inotify
```

### High Latency

**Symptoms:** Transcriptions are slow

**Solutions:**
- Check internet speed (for Deepgram)
- Try smaller Whisper model: `export STT_MODEL_SIZE="tiny"`
- Verify audio is being captured continuously
- Check CPU usage

## Future Enhancements

### Planned Features

1. **JACK Audio Capture** (Next)
   - Direct audio capture from baresip
   - Eliminates file I/O completely
   - 50-100ms latency improvement
   - See `audio/jack_capture.py` for plan

2. **Streaming Whisper** (Future)
   - True streaming local transcription
   - Partial results with Whisper
   - 500-800ms latency
   - No internet required

3. **Additional Providers** (Future)
   - AssemblyAI
   - Google Speech-to-Text
   - Azure Speech Services
   - Custom models

## Documentation

- [DEEPGRAM_INTEGRATION.md](DEEPGRAM_INTEGRATION.md) - Detailed integration guide
- [MIGRATION_CHECKLIST.md](MIGRATION_CHECKLIST.md) - Step-by-step migration
- [audio/jack_capture.py](src/mr_sip/audio/jack_capture.py) - Future JACK implementation plan

## Support

For issues:
1. Check logs in MindRoot output
2. Review troubleshooting section above
3. Verify environment variables
4. Test with simple audio file
5. Check [DEEPGRAM_INTEGRATION.md](DEEPGRAM_INTEGRATION.md)

## License

Same as MindRoot SIP Plugin
