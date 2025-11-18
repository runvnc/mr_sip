# PySIP Process Isolation for Smooth Audio

## Overview

This implementation runs PySIP's SIP/RTP handling in a separate OS process to eliminate GIL contention and ensure smooth audio processing during Speech-to-Speech calls.

## Architecture

```
Main Process                          PySIP Process
────────────────────────────────────────────────────────────────
OpenAI S2S API                        
    ↓                                 
MindRoot Agent                        
    ↓                                 
SIP Session Manager                   
    ↓                                 
PySIPProcessProxy                    MindRootSIPBotS2S
    ↓                                     ↓
multiprocessing.Queue  ←──────────→  PySIP SipCall
(bidirectional)                           ↓
                                      RTP Handler
```

## Key Benefits

✅ **No GIL Contention**: Separate Python interpreter per process
✅ **True Parallelism**: Audio processing runs independently
✅ **Smooth Audio**: No blocking from main event loop
✅ **Minimal Code Changes**: ~700 lines of new code, ~50 lines modified
✅ **Backwards Compatible**: Can toggle on/off via configuration

## Files Added

1. **`pysip_process_wrapper.py`** (471 lines)
   - Manages PySIP subprocess lifecycle
   - Handles multiprocessing queues
   - Monitors subprocess health

2. **`pysip_process_proxy.py`** (186 lines)
   - Provides same interface as MindRootSIPBotS2S
   - Forwards operations to subprocess
   - Transparent to rest of codebase

## Files Modified

1. **`sip_client_s2s.py`** (~50 lines changed)
   - Added optional queue parameters
   - Queue mode vs direct mode
   - Helper method for subprocess

2. **`services_s2s.py`** (~80 lines changed)
   - Process isolation by default
   - Environment variable override
   - Backwards compatible direct mode

## Configuration

### Enable/Disable Process Isolation

**Via Environment Variable** (recommended):
```bash
# Enable process isolation (default)
export SIP_USE_PROCESS_ISOLATION=true

# Disable process isolation (use direct mode)
export SIP_USE_PROCESS_ISOLATION=false
```

**Via Service Call**:
```python
await dial_service(
    destination="+15551234567",
    context=context,
    use_process_isolation=True  # or False
)
```

### Other Configuration

All existing SIP configuration still applies:
```bash
SIP_GATEWAY=sip.example.com:5060
SIP_USER=your_username
SIP_PASSWORD=your_password
SIP_ENABLE_RECORDING=true
SIP_RECORDING_DIR=data/calls
SIP_RECORD_SEPARATE=false
```

## How It Works

### Call Flow

1. **Main Process**:
   - Agent receives dial command
   - Creates `PySIPProcessWrapper`
   - Creates `PySIPProcessProxy`
   - Spawns PySIP subprocess

2. **PySIP Subprocess**:
   - Creates `MindRootSIPBotS2S` in queue mode
   - Initiates SIP call
   - Handles RTP audio
   - Sends status updates via queue

3. **Audio Flow**:
   - **Phone → OpenAI**: PySIP subprocess → audio_in_queue → Proxy → OpenAI
   - **OpenAI → Phone**: OpenAI → Proxy → audio_out_queue → PySIP subprocess

### Queue Management

- **Bounded Queues**: 200 frames max (~4 seconds)
- **Non-blocking Operations**: Drops frames if queue full
- **Low Latency**: Small queues minimize delay

### Process Lifecycle

1. **Startup**: Subprocess spawned on dial
2. **Running**: Handles audio until call ends
3. **Shutdown**: Clean termination with timeout
4. **Cleanup**: Queues drained, resources freed

## Monitoring

### Metrics

Get metrics from the proxy:
```python
metrics = bot.get_metrics()
print(metrics)
```

Output:
```python
{
    'running': True,
    'call_established': True,
    'audio_in_count': 1234,
    'audio_out_count': 5678,
    'audio_in_queue_size': 5,
    'audio_out_queue_size': 3,
    'process_alive': True,
    'uptime': 45.2
}
```

### Logging

Process isolation adds these log prefixes:
- `[PySIP-{log_id}]` - Subprocess logs
- `Proxy:` - Proxy operation logs
- `PROCESS ISOLATION mode` - Mode indicator

## Troubleshooting

### Audio Still Not Smooth

1. **Check CPU Usage**: Ensure system isn't overloaded
2. **Check Queue Sizes**: Monitor `audio_*_queue_size` metrics
3. **Increase Queue Size**: Edit `maxsize` in `pysip_process_wrapper.py`
4. **Check Network**: SIP/RTP issues can cause problems

### Process Won't Start

1. **Check Logs**: Look for subprocess errors
2. **Test Direct Mode**: Try `use_process_isolation=False`
3. **Check Permissions**: Ensure process can be spawned
4. **Check Dependencies**: Verify PySIP is installed

### Process Won't Stop

1. **Check Timeout**: Default is 5 seconds
2. **Force Kill**: Process will be killed if timeout exceeded
3. **Check Logs**: Look for cleanup errors

## Performance Tuning

### Queue Sizes

Edit `pysip_process_wrapper.py`:
```python
self.audio_in_queue = mp.Queue(maxsize=200)   # Increase for more buffering
self.audio_out_queue = mp.Queue(maxsize=200)  # Decrease for lower latency
```

**Trade-offs**:
- Larger queues = more buffering, higher latency, fewer drops
- Smaller queues = less buffering, lower latency, more drops

### CPU Affinity (Optional)

To pin PySIP to a specific CPU core (Linux only):

Edit `pysip_process_wrapper.py`, in `_run_pysip_process()`:
```python
import os
try:
    os.sched_setaffinity(0, {1})  # Pin to core 1
    logger.info("PySIP pinned to CPU core 1")
except AttributeError:
    pass  # Not supported on this platform
```

**Note**: Usually not necessary - OS scheduler is smart enough.

## Testing

### Test Process Isolation

```python
# Test with process isolation
result = await dial_service(
    destination="+15551234567",
    context=context,
    use_process_isolation=True
)
assert result['mode'] == 's2s_pysip_isolated'

# Test direct mode
result = await dial_service(
    destination="+15551234567",
    context=context,
    use_process_isolation=False
)
assert result['mode'] == 's2s_pysip_direct'
```

### Monitor Audio Quality

```python
# During call, check metrics
metrics = bot.get_metrics()
if metrics['audio_out_queue_size'] > 100:
    logger.warning("Audio queue backing up!")
```

## Migration Guide

### From Direct Mode

No changes needed! Process isolation is enabled by default.

To explicitly enable:
```bash
export SIP_USE_PROCESS_ISOLATION=true
```

### To Direct Mode

If you need to disable process isolation:
```bash
export SIP_USE_PROCESS_ISOLATION=false
```

Or per-call:
```python
await dial_service(
    destination="+15551234567",
    context=context,
    use_process_isolation=False
)
```

## Known Limitations

1. **Startup Latency**: ~100ms overhead to spawn process
2. **Memory Overhead**: ~10MB per subprocess
3. **Platform Support**: Tested on Linux/macOS, should work on Windows
4. **Queue Marshalling**: Small latency from pickling audio data

## Future Improvements

- [ ] Shared memory for audio (zero-copy)
- [ ] Process pooling (pre-spawn processes)
- [ ] Dynamic queue sizing based on load
- [ ] CPU affinity configuration
- [ ] Real-time priority for audio process

## Support

For issues or questions:
1. Check logs for errors
2. Try direct mode to isolate issue
3. Monitor metrics for queue health
4. Review this documentation

## Technical Details

### Why Multiprocessing?

- **GIL**: Python's Global Interpreter Lock prevents true parallelism in threads
- **Separate Interpreter**: Each process has its own GIL
- **True Parallelism**: Audio processing doesn't block main event loop

### Why Not Threading?

- Threads share GIL - no benefit for CPU-bound work
- Audio processing is CPU-bound (encoding, buffering, etc.)
- Threading would still have contention issues

### Why Not Async?

- Async is cooperative - still single-threaded
- Blocking operations (like audio encoding) block event loop
- Can't achieve true parallelism with async alone

### Queue vs Shared Memory?

- Queues are simpler and more reliable
- Shared memory is faster but more complex
- For audio, queue overhead is negligible (~1-2ms)
- Future optimization: can switch to shared memory if needed

## Conclusion

Process isolation provides smooth audio by eliminating GIL contention with minimal code changes. It's enabled by default and transparent to the rest of the codebase.

For most use cases, the default configuration works well. Advanced users can tune queue sizes and enable CPU affinity for specific requirements.
