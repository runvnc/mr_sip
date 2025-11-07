# Voice AI Pipeline Latency Optimization - Implementation Summary

**Date:** 2025-11-06
**Status:** ✅ COMPLETE - All Phases 1-3 Implemented

---

## Overview

Successfully implemented latency optimizations targeting 220-260ms reduction in end-to-end latency through buffer optimization and Opus codec enablement. Phase 4 (VPS migration) requires infrastructure changes and is documented separately.

---

## Changes Implemented

### Phase 1: Quick Wins (~150ms savings) ✅

#### 1.1 Reduced Audio Chunk Size
**File:** `src/mr_sip/sip_client_s2s.py` (line ~95)

**Change:**
- `chunk_duration_s`: 0.1 → 0.020 (100ms → 20ms chunks)
- **Expected savings:** 80ms on input path

#### 1.2 Disabled AGC
**File:** `src/mr_sip/sip_client_s2s.py` (line ~98)

**Change:**
- `agc_target_rms`: 0.15 → 0.0 (AGC disabled)
- **Rationale:** Phone audio from Telnyx is already normalized
- **Expected savings:** 50-100ms processing time

#### 1.3 Added Output Queue Limit
**File:** `src/mr_sip/sip_manager.py` (line ~27)

**Changes:**
- Added `maxsize=10` to audio queue (limits to ~200ms of audio)
- Added timeout handling with 0.1s timeout in `send_audio()` method
- Drops chunks when queue is full to prevent latency accumulation
- **Expected savings:** Prevents 100-500ms latency spikes

---

### Phase 2: JACK Optimization (~30-50ms savings) ✅

#### 2.1 Reduced JACK Buffer Size
**File:** `src/mr_sip/start_jack_daemon.sh` (lines 11-13)

**Changes:**
- `PERIOD_SIZE`: 256 → 128 frames
- `WAIT_TIME`: 32000 → 16000 microseconds
- **At 8kHz:** 32ms → 16ms buffer latency
- **Expected savings:** 16ms on both input and output paths (32ms total)

---

### Phase 3: Opus Codec Enablement (~40-60ms savings) ✅

#### 3.1 Updated JACK to 24kHz
**File:** `src/mr_sip/start_jack_daemon.sh` (lines 11-13)

**Changes:**
- `SAMPLE_RATE`: 8000 → 24000 Hz
- `PERIOD_SIZE`: 128 frames (now 5.3ms at 24kHz)
- `WAIT_TIME`: 16000 → 5333 microseconds
- **Benefits:**
  - Matches OpenAI Realtime API (24kHz)
  - Eliminates resampling overhead: 10-20ms
  - Smaller buffer at 24kHz: additional 10-15ms
  - **Expected savings:** 20-35ms total

#### 3.2 Added Opus Module Loading
**File:** `src/mr_sip/audio_handler.py` (line ~33)

**Changes:**
- Added Opus module loading in `configure_baresip_jack()`
- Graceful fallback if Opus module not available
- Logs success/failure for monitoring

#### 3.3 Added Codec Verification Logging
**File:** `src/mr_sip/sip_client_s2s.py` (line ~71)

**Changes:**
- Added codec status check in `handle_call_established()`
- Logs active codec for verification
- Helps confirm Opus negotiation success

---

## Total Expected Latency Reduction

| Phase | Latency Savings | Status |
|-------|----------------|--------|
| Phase 1: Quick Wins | ~150ms | ✅ Complete |
| Phase 2: JACK Optimization | ~30-50ms | ✅ Complete |
| Phase 3: Opus Enablement | ~40-60ms | ✅ Complete |
| **TOTAL (Phases 1-3)** | **220-260ms** | **✅ Complete** |

---

## Next Steps

### 1. Restart JACK Daemon

```bash
# Stop existing JACK
killall jackd

# Start with new settings
cd /xfiles/update_plugins/mr_sip/src/mr_sip
./start_jack_daemon.sh

# Verify JACK is running at 24kHz
jack_samplerate
# Should output: 24000
```

### 2. Restart Application

Restart your MindRoot SIP application to pick up the code changes.

### 3. Configure Baresip for Opus (Optional but Recommended)

**File:** `~/.baresipy/config`

Add or modify these lines:

```ini
# Audio Codecs (order = priority)
audio_codecs           opus/24000/1
audio_codecs           PCMU/8000/1
audio_codecs           PCMA/8000/1

# Opus Settings
opus_bitrate           32000      # 32 kbps (good for voice)
opus_complexity        5          # 0-10, lower = faster encoding
opus_packet_loss       10         # Expected packet loss %
opus_fec               yes        # Enable Forward Error Correction
```

**Note:** If `opus/24000/1` is not supported, try `opus/48000/2`

### 4. Configure Telnyx for Opus

1. Log into Telnyx portal
2. Navigate to: **Voice → SIP Connections → [Your Connection]**
3. Find **Codecs** or **Audio Settings** section
4. Enable and prioritize:
   - opus (highest priority)
   - PCMU (G.711 μ-law fallback)
   - PCMA (G.711 A-law fallback)

### 5. Test and Monitor

#### Make Test Call
```bash
# Monitor JACK for xruns (audio glitches)
tail -f /tmp/jackd.log | grep -i xrun

# Monitor application logs for:
# - "Active codec: opus" (confirms Opus is working)
# - "Audio queue full" warnings (should be rare)
# - No audio glitches or dropouts
```

#### Success Criteria
- [ ] JACK running at 24kHz (verify with `jack_samplerate`)
- [ ] No xruns in JACK log during test calls
- [ ] Codec logs show "opus" when available
- [ ] Noticeable improvement in response time
- [ ] No audio quality degradation
- [ ] Minimal "Audio queue full" warnings (<1% of chunks)

---

## Monitoring Key Metrics

### 1. JACK Status
```bash
# Check sample rate
jack_samplerate

# Check buffer size
jack_bufsize

# List ports and connections
jack_lsp -c

# Monitor for xruns
tail -f /tmp/jackd.log | grep -i xrun
```

### 2. Application Logs
Look for:
- `Active codec: opus/24000/1` (or opus/48000/2)
- `Loaded Opus codec module`
- `Audio queue full` warnings (should be rare)
- No resampling messages (eliminated with 24kHz JACK)

### 3. Audio Quality
- Listen for improved responsiveness
- Check for any audio glitches or dropouts
- Verify natural conversation flow

---

## Rollback Procedures

### If Phase 1 Causes Issues

**File:** `src/mr_sip/sip_client_s2s.py`
```python
# Revert chunk size
chunk_duration_s=0.1

# Re-enable AGC
agc_target_rms=0.15
```

**File:** `src/mr_sip/sip_manager.py`
```python
# Remove queue limit
self.audio_queue = asyncio.Queue()

# Remove timeout in send_audio()
await self.audio_queue.put(audio_chunk)
```

### If Phase 2 Causes Issues

**File:** `src/mr_sip/start_jack_daemon.sh`
```bash
# Revert JACK buffer
PERIOD_SIZE=256
WAIT_TIME=32000
```

### If Phase 3 Causes Issues

**File:** `src/mr_sip/start_jack_daemon.sh`
```bash
# Revert JACK to 8kHz
SAMPLE_RATE=8000
PERIOD_SIZE=128  # or 256
WAIT_TIME=16000  # or 32000
```

**File:** `~/.baresipy/config`
```ini
# Remove opus, keep only:
audio_codecs           PCMU/8000/1
audio_codecs           PCMA/8000/1
```

---

## Phase 4: VPS Migration (Not Yet Implemented)

**Expected savings:** ~240ms (network latency reduction)

This phase requires:
1. Provisioning a VPS in Virginia (AWS us-east-1, DigitalOcean NYC3, etc.)
2. Deploying the application to the VPS
3. Configuring the VPS environment
4. Testing and migration

See `LATENCY_OPTIMIZATION_PLAN.md` Phase 4 for detailed instructions.

---

## Files Modified

1. `/xfiles/update_plugins/mr_sip/src/mr_sip/sip_client_s2s.py`
   - Reduced chunk size to 20ms
   - Disabled AGC
   - Added codec verification logging

2. `/xfiles/update_plugins/mr_sip/src/mr_sip/sip_manager.py`
   - Added queue size limit (maxsize=10)
   - Added timeout handling in send_audio()

3. `/xfiles/update_plugins/mr_sip/src/mr_sip/start_jack_daemon.sh`
   - Reduced PERIOD_SIZE to 128
   - Updated SAMPLE_RATE to 24000 Hz
   - Adjusted WAIT_TIME accordingly

4. `/xfiles/update_plugins/mr_sip/src/mr_sip/audio_handler.py`
   - Added Opus module loading

---

## Troubleshooting

### Issue: JACK xruns appearing
**Solution:** Increase PERIOD_SIZE back to 256 in `start_jack_daemon.sh`

### Issue: Audio glitches or dropouts
**Solution:** 
1. Check CPU usage during calls
2. Increase PERIOD_SIZE if needed
3. Verify JACK is running properly

### Issue: "Audio queue full" warnings frequent
**Solution:**
1. This is expected behavior to prevent latency buildup
2. If >1% of chunks are dropped, investigate:
   - Network issues
   - CPU bottlenecks
   - JACK configuration

### Issue: Opus not negotiating
**Solution:**
1. Check baresip config has opus enabled
2. Verify Telnyx has opus enabled
3. Check logs for "Loaded Opus codec module"
4. Try opus/48000/2 if opus/24000/1 doesn't work

### Issue: No improvement in latency
**Solution:**
1. Verify JACK is running at 24kHz: `jack_samplerate`
2. Check logs for codec in use
3. Ensure application was restarted after code changes
4. Monitor for "Audio queue full" - if none, queue limit may not be helping

---

## Performance Expectations

### Before Optimization
- Input path: ~150-200ms
- Output path: ~150-200ms
- Total buffering overhead: ~300-400ms

### After Optimization (Phases 1-3)
- Input path: ~70-90ms (60-110ms reduction)
- Output path: ~70-90ms (60-110ms reduction)
- Total buffering overhead: ~140-180ms (160-220ms reduction)

### With Phase 4 (VPS Migration)
- Additional network latency reduction: ~240ms
- **Total expected improvement: 400-460ms**

---

## Notes

1. **AGC Disabled:** Phone audio from Telnyx is already normalized, so AGC adds latency without much benefit. If you notice audio level issues, you can re-enable AGC with a reduced window (0.1s instead of 1.5s).

2. **Queue Drops:** The "Audio queue full" warnings are intentional - they prevent latency accumulation by dropping chunks when the system can't keep up. This is preferable to building up a large delay.

3. **Opus Codec:** Opus provides better audio quality and lower latency than G.711. However, it requires support from both baresip and Telnyx. The system will fall back to G.711 if Opus negotiation fails.

4. **24kHz JACK:** Running JACK at 24kHz eliminates resampling overhead since OpenAI Realtime API uses 24kHz. This is a significant optimization.

5. **Testing:** Test thoroughly after each phase. If issues arise, roll back that phase and investigate before proceeding.

---

## Success! 🎉

All code changes for Phases 1-3 have been successfully implemented. The system is now optimized for:
- ✅ Reduced buffer latency (20ms chunks, 128-frame JACK buffer)
- ✅ Eliminated AGC overhead
- ✅ Prevented queue buildup with size limits and timeouts
- ✅ Matched sample rates (24kHz) to eliminate resampling
- ✅ Opus codec support for better quality and lower latency

**Next:** Restart JACK, restart the application, and test!
