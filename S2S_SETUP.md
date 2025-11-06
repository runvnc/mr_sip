# Speech-to-Speech (S2S) Mode Setup Guide

This guide explains how to configure and use the MindRoot SIP plugin in Speech-to-Speech mode with OpenAI Realtime API.

## Overview

In S2S mode, the architecture is:

```
Phone Call → JACK → SIP Client → OpenAI Realtime API
                                        ↓
                                  (audio + commands)
                                        ↓
Phone Call ← JACK ← sip_audio_out_chunk ← SpeechToSpeechAgent
```

## Key Features

- **Integrated Pipeline**: OpenAI handles STT + LLM + TTS in one stream
- **Lower Latency**: Fewer round trips than separate STT/TTS
- **Agent Control**: Agent manages workflow and audio routing
- **Flexible**: Agent can work standalone or with phone calls

## Configuration

### 1. Environment Variables

Create or update your `.env` file:

```bash
# ============================================
# Mode Selection
# ============================================
SIP_PROVIDER=s2s  # Use 's2s' for Speech-to-Speech mode

# ============================================
# SIP Configuration
# ============================================
SIP_GATEWAY=your.sip.gateway.com
SIP_USER=your_sip_username
SIP_PASSWORD=your_sip_password
AUDIO_CAPTURE_METHOD=jack  # Must be jack for S2S mode
SIP_CALL_ESTABLISH_TIMEOUT=120  # Timeout in seconds

# ============================================
# OpenAI Configuration
# ============================================
OPENAI_API_KEY=sk-...
OPENAI_REALTIME_MODEL=gpt-4o-realtime-preview-2024-10-01
OPENAI_VOICE=alloy  # Options: alloy, echo, fable, onyx, nova, shimmer
```

### 2. Agent Configuration

Create an agent that uses `SpeechToSpeechAgent` class:

```json
{
  "name": "PhoneAgent",
  "agent_class": "SpeechToSpeechAgent",
  "model": "gpt-4o-realtime-preview-2024-10-01",
  "commands": [
    "call",
    "hangup",
    "task_result"
  ],
  "system_prompt": "You are a helpful phone assistant..."
}
```

### 3. System Instructions

The agent's system prompt should include instructions for using phone commands:

```
You can make phone calls using the 'call' command:
{ "call": { "destination": "1234567890" } }

To end a call:
{ "hangup": {} }

When you're done with your task:
{ "task_result": { "output": "Summary of what was accomplished" } }
```

## Usage Flow

### Example Conversation

1. **Agent starts locally**
   - User: "Please call John at 555-1234 and ask about the meeting"
   - Agent can test audio locally first

2. **Agent makes call**
   - Agent executes: `{ "call": { "destination": "5551234" } }`
   - SIP call established
   - Audio routes to phone automatically

3. **Conversation happens**
   - Caller speaks → captured via JACK → sent to OpenAI
   - OpenAI responds → audio routed to phone via agent
   - Agent can execute commands during call

4. **Agent ends call**
   - Agent executes: `{ "hangup": {} }`
   - Audio returns to local playback

5. **Agent completes task**
   - Agent executes: `{ "task_result": { "output": "Called John, meeting confirmed for 3pm" } }`

## Architecture Details

### Components

1. **SpeechToSpeechAgent** (`/files/mindroot/src/mindroot/coreplugins/agent/speech_to_speech.py`)
   - Manages OpenAI Realtime session
   - Routes audio output based on call state
   - Executes commands (call, hangup, etc.)

2. **MindRootSIPBotS2S** (`/xfiles/update_plugins/mr_sip/src/mr_sip/sip_client_s2s.py`)
   - Handles SIP call lifecycle
   - Captures audio from JACK
   - Sends audio to OpenAI via `send_s2s_audio_chunk`

3. **S2S Services** (`/xfiles/update_plugins/mr_sip/src/mr_sip/services_s2s.py`)
   - `dial_service`: Initiates SIP calls
   - `end_call_service`: Terminates calls
   - `sip_audio_out_chunk`: Routes audio to phone (reused from base services)

### Audio Flow

**Input (Caller speaks):**
```
Phone → SIP → baresip → JACK → JACKAudioCapture
  → send_s2s_audio_chunk → OpenAI Realtime API
```

**Output (Agent responds):**
```
OpenAI Realtime API → on_audio_chunk callback
  → SpeechToSpeechAgent.on_audio_chunk_callback
  → (if on call) sip_audio_out_chunk
  → JACK → baresip → SIP → Phone
```

## Troubleshooting

### JACK Issues

1. **Check JACK is running:**
   ```bash
   pgrep -x jackd
   ```

2. **View JACK logs:**
   ```bash
   cat /tmp/mr_sip_logs/jack_startup.log
   ```

3. **Manually start JACK:**
   ```bash
   cd /xfiles/update_plugins/mr_sip/src/mr_sip
   ./start_jack_daemon.sh
   ```

### Audio Issues

1. **No audio from caller:**
   - Check JACK port connections
   - Verify baresip is using JACK for auplay
   - Check audio capture logs

2. **No audio to caller:**
   - Verify `on_sip_call` flag is set correctly
   - Check `sip_audio_out_chunk` is being called
   - Verify JACK output connections

3. **Audio quality issues:**
   - Check sample rate conversions (24kHz for OpenAI)
   - Verify AGC settings in JACKAudioCapture
   - Check network latency

### Call Issues

1. **Call fails to establish:**
   - Verify SIP credentials
   - Check SIP gateway connectivity
   - Increase `SIP_CALL_ESTABLISH_TIMEOUT`

2. **Call drops unexpectedly:**
   - Check network stability
   - Review baresip logs
   - Verify JACK audio is flowing

### OpenAI Issues

1. **No response from OpenAI:**
   - Verify API key is valid
   - Check OpenAI service status
   - Review websocket connection logs

2. **Commands not executing:**
   - Verify agent has commands enabled
   - Check command format in system prompt
   - Review `handle_s2s_cmd` logs

## Testing

### Test Local Audio First

```python
# Start agent without making a call
# Audio should play locally
# Verify you can hear the agent
```

### Test Call Establishment

```python
# Have agent execute call command
# Verify call connects
# Check JACK connections are made
```

### Test Audio Quality

```python
# Speak into phone
# Verify agent hears and responds
# Check for latency and clarity
```

### Test Call Termination

```python
# Have agent execute hangup
# Verify call ends cleanly
# Check audio returns to local
```

## Performance Tuning

### Latency Optimization

1. **Reduce chunk duration:**
   ```python
   chunk_duration_s=0.05  # In JACKAudioCapture
   ```

2. **Adjust JACK buffer size:**
   ```bash
   # In start_jack_daemon.sh
   -p 128  # Smaller buffer = lower latency
   ```

3. **Network optimization:**
   - Use wired connection
   - Minimize network hops
   - Consider QoS settings

### Audio Quality

1. **AGC settings:**
   ```python
   agc_target_rms=0.15  # Target volume level
   agc_max_gain=20.0    # Maximum amplification
   ```

2. **Sample rate:**
   - OpenAI expects 24kHz
   - JACK typically runs at 8kHz or 48kHz
   - Resampling is automatic

## Comparison with Deepgram Mode

| Feature | S2S Mode | Deepgram Mode |
|---------|----------|---------------|
| STT Provider | OpenAI | Deepgram |
| TTS Provider | OpenAI | ElevenLabs |
| Latency | Lower (integrated) | Higher (separate) |
| Setup | Simpler | More complex |
| Cost | OpenAI pricing | Deepgram + ElevenLabs |
| Flexibility | Less (one provider) | More (mix providers) |

## Advanced Usage

### Multiple Calls in Sequence

The agent can make multiple calls in one session:

```
1. Start agent
2. Call person A
3. Conversation
4. Hangup
5. Call person B
6. Conversation
7. Hangup
8. Complete task
```

### Database Integration

Agent can query databases during calls:

```json
{
  "commands": [
    "call",
    "hangup",
    "query_database",
    "task_result"
  ]
}
```

### Custom Commands

Add plugin-specific commands:

```python
@command()
async def transfer_call(destination: str, context=None):
    # Implementation
    pass
```

## Security Considerations

1. **API Keys**: Store securely in `.env`, never commit
2. **SIP Credentials**: Use strong passwords
3. **Network**: Use encrypted connections when possible
4. **Logging**: Be careful with PII in logs

## Future Enhancements

- Support for other S2S providers (Anthropic, etc.)
- Call recording and transcription storage
- Multi-party conference calls
- Call transfer and forwarding
- IVR menu navigation

## Support

For issues or questions:
1. Check logs in `/tmp/mr_sip_logs/`
2. Review this documentation
3. Check the main plan document: `/tmp/s2splan.md`
4. File an issue in the repository
