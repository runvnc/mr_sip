# Deepgram Required Mode

## Overview

By default, the mr_sip plugin now **requires Deepgram** and will exit with a clear error message if it's not properly configured.

## Default Behavior

### ✅ Deepgram Required (Default)

```bash
# Default: Deepgram is required
# If not configured, the process will exit with a clear error
```

**What happens:**
1. Plugin loads
2. Checks for `STT_PROVIDER=deepgram`
3. Checks for `DEEPGRAM_API_KEY`
4. Tests Deepgram connection
5. If any check fails: **Process exits with detailed error message**

### Error Messages You'll See

#### Missing API Key

```
================================================================================
FATAL ERROR: DEEPGRAM_API_KEY environment variable not set
================================================================================
Deepgram is required but no API key was provided.

To fix this:
1. Get an API key from https://deepgram.com/
2. Set it: export DEEPGRAM_API_KEY='your_key_here'

Or to disable this requirement:
   export REQUIRE_DEEPGRAM=false
================================================================================
```

#### Wrong STT Provider

```
================================================================================
FATAL ERROR: Deepgram is required but STT_PROVIDER='whisper_vad'
================================================================================
Please set: export STT_PROVIDER=deepgram
Or disable requirement: export REQUIRE_DEEPGRAM=false
================================================================================
```

#### Connection Failed

```
================================================================================
❌ FATAL ERROR: Failed to connect to Deepgram
================================================================================
Error: [connection error details]

Possible causes:
1. Invalid API key
2. No internet connection
3. Firewall blocking WebSocket connections
4. Deepgram service is down

Please verify your DEEPGRAM_API_KEY and internet connection.
================================================================================
```

## Configuration

### Correct Setup (Required)

```bash
# Set these environment variables
export STT_PROVIDER="deepgram"
export DEEPGRAM_API_KEY="your_deepgram_api_key_here"

# Optional: explicitly enable requirement (already default)
export REQUIRE_DEEPGRAM=true
```

### Disable Requirement (Not Recommended)

If you want to use Whisper instead:

```bash
# Disable Deepgram requirement
export REQUIRE_DEEPGRAM=false

# Use Whisper instead
export STT_PROVIDER="whisper_vad"
export STT_MODEL_SIZE="small"
```

## Startup Logging

### Successful Initialization

You'll see these messages when everything is configured correctly:

```
INFO: SIP Plugin Configuration: V2=enabled, STT_PROVIDER=deepgram, REQUIRE_DEEPGRAM=True

================================================================================
INITIALIZING DEEPGRAM STT PROVIDER
================================================================================
API Key: 1234567890...abcd
Destination: 16822625850
Session: abc123xyz
================================================================================

INFO: Testing Deepgram connection...

================================================================================
✅ DEEPGRAM CONNECTION SUCCESSFUL
================================================================================

INFO: Call established to 16822625850. (Using deepgram for transcription)
```

### Failed Initialization

If something is wrong, you'll see:

```
INFO: SIP Plugin Configuration: V2=enabled, STT_PROVIDER=deepgram, REQUIRE_DEEPGRAM=True

ERROR: [Detailed error message as shown above]

[Process exits]
```

## Why This Approach?

### Benefits

1. **Fail Fast** - Errors are caught immediately, not during a call
2. **Clear Messages** - Detailed instructions on how to fix the problem
3. **No Silent Failures** - Can't accidentally use wrong STT provider
4. **Production Ready** - Ensures Deepgram is working before accepting calls

### When to Disable

Only disable `REQUIRE_DEEPGRAM` if:
- You're in development and want to test without Deepgram
- You explicitly want to use Whisper VAD instead
- You're troubleshooting and need to bypass the check

## Testing Your Setup

### Step 1: Set Environment Variables

```bash
export STT_PROVIDER="deepgram"
export DEEPGRAM_API_KEY="your_key_here"
```

### Step 2: Start MindRoot

```bash
# Start your MindRoot instance
# Watch for the initialization messages
```

### Step 3: Look for Success Message

You should see:

```
✅ DEEPGRAM CONNECTION SUCCESSFUL
```

If you see this, you're good to go!

### Step 4: Make a Test Call

```python
{ "call": { "destination": "your_test_number" } }
```

## Troubleshooting

### Problem: Process exits immediately

**Check:**
1. Is `DEEPGRAM_API_KEY` set? `echo $DEEPGRAM_API_KEY`
2. Is `STT_PROVIDER` set to `deepgram`? `echo $STT_PROVIDER`
3. Do you have internet connection? `ping api.deepgram.com`

### Problem: "Invalid API key" error

**Solution:**
1. Verify your API key at https://deepgram.com/
2. Make sure you copied it correctly (no extra spaces)
3. Try creating a new API key

### Problem: "Connection failed" error

**Solution:**
1. Check internet connection
2. Check firewall settings (allow WebSocket connections)
3. Try: `curl -H "Authorization: Token $DEEPGRAM_API_KEY" https://api.deepgram.com/v1/projects`

### Problem: Want to use Whisper instead

**Solution:**
```bash
export REQUIRE_DEEPGRAM=false
export STT_PROVIDER="whisper_vad"
export STT_MODEL_SIZE="small"
```

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `REQUIRE_DEEPGRAM` | `true` | Require Deepgram or exit |
| `STT_PROVIDER` | `deepgram` | STT provider to use |
| `DEEPGRAM_API_KEY` | (none) | Your Deepgram API key |
| `SIP_USE_V2` | `true` | Use V2 implementation |

## Summary

✅ **Deepgram is required by default**
✅ **Clear error messages if misconfigured**
✅ **Connection tested before accepting calls**
✅ **Process exits if Deepgram unavailable**
✅ **Can be disabled with `REQUIRE_DEEPGRAM=false`**

This ensures production reliability and prevents silent failures!
