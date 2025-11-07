# PySIP S2S Migration Plan - Compact Engineering Handoff

**Date**: 2025-11-07  
**Goal**: Replace baresip/JACK with PySIP for S2S mode  
**Key Insight**: OpenAI Realtime API supports ulaw 8kHz directly - NO conversion needed!

## Critical Finding: send_tts_audio Must Be Preserved

The `send_tts_audio()` method is called by:
1. `sip_manager.py` - `SIPSession._send_audio_to_sip()` 
2. `services.py` - `sip_audio_out_chunk()` service (via session manager)

**This interface MUST be maintained** - it's how audio gets routed from OpenAI to the phone.

## PySIP Audio Sending API

From `/files/PySIP/PySIP/rtp_handler.py`:

```python
# Audio is sent via RTP session's audio stream mechanism:
self._rtp_session.set_audio_stream(audio_stream)

# Where audio_stream is an AudioStream object that:
# - Has an input_q (queue.Queue) that feeds frames
# - Encoder reads from this queue and sends via RTP
```

**Key Pattern**: PySIP uses `AudioStream` objects with queues, not direct byte sending.

## Files to Modify

### 1. `/xfiles/update_plugins/mr_sip/src/mr_sip/sip_client_s2s.py`

**Current**: Inherits from `BareSIP`, uses JACK for audio I/O  
**New**: Standalone class using PySIP `SipCall`

**Key Changes**:
- Remove: `BareSIP` inheritance, all JACK code, `audio_handler` usage
- Add: PySIP `SipCall` instance, RTP callbacks
- Keep: `send_tts_audio()` method signature (required by session manager)

**Implementation**:

```python
from PySIP import SipCall
from PySIP.filters import CallState
import asyncio
import queue
from datetime import datetime

class MindRootSIPBotS2S:
    def __init__(self, user, password, gateway, context=None):
        self.sip_username = user
        self.sip_password = password
        self.sip_server = gateway
        self.context = context
        
        self.call = None
        self.is_active = False
        self.call_established = False
        self.call_start_time = None
        self.audio_output_queue = None  # For sending audio to RTP
        
    async def make_call(self, destination):
        """Initiate outbound call."""
        self.call = SipCall(
            username=self.sip_username,
            password=self.sip_password,
            route=self.sip_server,
            callee=destination
        )
        
        # Setup callbacks
        @self.call.on_call_state_changed
        async def on_state(state):
            if state == CallState.ANSWERED:
                await self._on_call_answered()
            elif state in [CallState.ENDED, CallState.FAILED, CallState.BUSY]:
                await self._on_call_ended()
        
        @self.call.on_frame_received
        async def on_frame(frame: bytes):
            # frame is ulaw 8kHz - send directly to OpenAI!
            await service_manager.send_s2s_audio_chunk(
                audio_bytes=frame,
                context=self.context
            )
        
        await self.call.start()
    
    async def _on_call_answered(self):
        """Called when call connects."""
        self.is_active = True
        self.call_established = True
        self.call_start_time = datetime.now()
        
        # Create queue for audio output
        self.audio_output_queue = queue.Queue()
        
        # Start audio output feeder task
        asyncio.create_task(self._audio_output_feeder())
    
    async def _on_call_ended(self):
        """Called when call ends."""
        self.is_active = False
        self.call_established = False
        
        # Stop audio output
        if self.audio_output_queue:
            self.audio_output_queue.put(None)
        
        # Send disconnect message
        await self._show_disconnected()
    
    async def _audio_output_feeder(self):
        """Feed audio from queue to PySIP RTP stream."""
        from PySIP.audio_stream import AudioStream
        
        while self.is_active:
            try:
                # Create AudioStream that reads from our queue
                audio_stream = AudioStream(self.audio_output_queue)
                
                # Set it on the RTP session
                if self.call._rtp_session:
                    self.call._rtp_session.set_audio_stream(audio_stream)
                
                # Wait for stream to finish
                await audio_stream.wait_finished()
                
            except Exception as e:
                logger.error(f"Error in audio output feeder: {e}")
                await asyncio.sleep(0.1)
    
    async def send_tts_audio(self, audio_chunk: bytes):
        """Send audio to call (REQUIRED by session manager).
        
        Args:
            audio_chunk: ulaw 8kHz audio from OpenAI
        """
        if not self.is_active or not self.audio_output_queue:
            logger.warning("Cannot send audio - call not active")
            return
        
        # Put audio chunk in queue for RTP sender
        # Note: PySIP expects 160-byte frames for 8kHz ulaw
        # May need to chunk the data appropriately
        try:
            await asyncio.to_thread(self.audio_output_queue.put, audio_chunk)
        except Exception as e:
            logger.error(f"Failed to queue audio: {e}")
    
    async def hangup_call(self):
        """Terminate the call."""
        if self.call:
            await self.call.stop("Agent hangup")
    
    async def _show_disconnected(self):
        """Send disconnect message to agent."""
        msg = "\n\nSYSTEM: -- CALL DISCONNECTED --\n\n"
        await service_manager.send_message_to_agent(
            session_id=self.context.log_id,
            message=msg,
            context=self.context
        )
```

**Audio Format Notes**:
- Input: PySIP `on_frame_received` gives ulaw 8kHz frames (typically 160 bytes)
- Output: OpenAI sends ulaw 8kHz - pass through directly
- No conversion needed!

### 2. `/xfiles/update_plugins/mr_sip/src/mr_sip/services_s2s.py`

**Changes**: Minimal - just update bot creation and call establishment

```python
@service()
async def dial_service(destination: str, context=None):
    # Create PySIP client
    bot = MindRootSIPBotS2S(
        user=SIP_USER,
        password=SIP_PASSWORD,
        gateway=SIP_GATEWAY,
        context=context
    )
    
    # Create session
    session_manager = get_session_manager()
    session = await session_manager.create_session(
        log_id=context.log_id,
        destination=destination,
        baresip_bot=bot  # Keep name for compatibility
    )
    
    # Make call (async - waits for answer)
    await bot.make_call(destination)
    
    # Wait for call to be established
    max_wait = CALL_ESTABLISH_TIMEOUT
    wait_count = 0
    while not bot.call_established and wait_count < max_wait:
        await asyncio.sleep(0.2)
        wait_count += 0.2
    
    if bot.call_established:
        session.is_active = True
        await session.start_audio_sender()
        return {
            "status": "call_established",
            "log_id": context.log_id,
            "destination": destination,
            "mode": "s2s"
        }
    else:
        await session_manager.end_session(context.log_id)
        return {
            "status": "call_failed",
            "log_id": context.log_id,
            "destination": destination,
            "error": "Call failed to establish"
        }
```

**Key**: Keep `baresip_bot` parameter name in session creation for compatibility with `sip_manager.py`.

### 3. `/xfiles/update_plugins/mr_sip/src/mr_sip/sip_manager.py`

**Changes**: NONE required! 

The session manager already calls `send_tts_audio()` which we're preserving in the new client.

```python
# This code stays exactly the same:
async def _send_audio_to_sip(self, audio_chunk: bytes):
    if self.baresip_bot and hasattr(self.baresip_bot, 'send_tts_audio'):
        await self.baresip_bot.send_tts_audio(audio_chunk)
```

## Files to Remove/Ignore

For S2S mode, these are no longer needed:
- `audio_handler.py` - No JACK, no conversion
- `audio/jack_input_capture.py` - No JACK
- `jack_streamer.py` - No JACK
- `start_jack_daemon.sh` - No JACK

**Note**: Don't delete these files as they're used by non-S2S modes. Just don't import/use them in S2S client.

## Implementation Checklist

1. **Backup current files**:
   ```bash
   cp src/mr_sip/sip_client_s2s.py src/mr_sip/sip_client_s2s.py.baresip.bak
   cp src/mr_sip/services_s2s.py src/mr_sip/services_s2s.py.baresip.bak
   ```

2. **Rewrite `sip_client_s2s.py`**:
   - Remove all JACK imports and code
   - Remove `BareSIP` inheritance
   - Implement PySIP pattern from above
   - **CRITICAL**: Keep `send_tts_audio()` method

3. **Update `services_s2s.py`**:
   - Change bot creation to use new client
   - Remove JACK-related waits
   - Keep session manager integration

4. **Test**:
   - Call establishment
   - Audio input (phone → OpenAI)
   - Audio output (OpenAI → phone)
   - Call termination
   - Multiple calls in sequence

## Open Questions to Resolve

### 1. Audio Frame Chunking

PySIP RTP expects 160-byte frames for 8kHz ulaw (20ms). OpenAI may send different chunk sizes.

**Solution**: In `send_tts_audio()`, chunk the incoming audio:

```python
async def send_tts_audio(self, audio_chunk: bytes):
    # Chunk into 160-byte frames
    FRAME_SIZE = 160
    for i in range(0, len(audio_chunk), FRAME_SIZE):
        frame = audio_chunk[i:i+FRAME_SIZE]
        if len(frame) == FRAME_SIZE:  # Only send complete frames
            await asyncio.to_thread(self.audio_output_queue.put, frame)
```

### 2. AudioStream Integration

Need to verify the exact pattern for feeding audio to PySIP's RTP session. The `AudioStream` class expects either:
- A file-like object (BytesIO)
- A queue that it reads from

**Approach**: Use queue-based feeding as shown in implementation above.

### 3. Call State Synchronization

Ensure `call_established` flag is set correctly for `services_s2s.py` to detect successful connection.

## Testing Strategy

1. **Unit Test**: Audio chunking logic
2. **Integration Test**: 
   - Make test call
   - Verify audio flows both ways
   - Check call termination
3. **Stress Test**: Multiple sequential calls
4. **Error Test**: Network failures, timeouts

## Rollback Plan

If issues arise:

```bash
# Quick rollback
cp src/mr_sip/sip_client_s2s.py.baresip.bak src/mr_sip/sip_client_s2s.py
cp src/mr_sip/services_s2s.py.baresip.bak src/mr_sip/services_s2s.py

# Restart JACK if needed
./src/mr_sip/start_jack_daemon.sh
```

## Estimated Time

- Implementation: 3-4 hours
- Testing: 2-3 hours
- Documentation: 1 hour
- **Total**: 6-8 hours

## Success Criteria

- ✅ Call establishment works
- ✅ Audio flows phone → OpenAI
- ✅ Audio flows OpenAI → phone
- ✅ No JACK dependencies
- ✅ No baresip dependencies
- ✅ `send_tts_audio()` interface preserved
- ✅ Session manager integration works
- ✅ Multiple calls work
- ✅ Clean error handling

## Key Takeaways

1. **Preserve `send_tts_audio()`** - Required by session manager
2. **No audio conversion** - OpenAI supports ulaw 8kHz directly
3. **PySIP uses AudioStream + queues** - Not direct byte sending
4. **Frame chunking may be needed** - 160 bytes per frame for 8kHz
5. **Session manager unchanged** - Just works with new client
