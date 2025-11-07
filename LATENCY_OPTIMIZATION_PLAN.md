# Voice AI Pipeline Latency Optimization Plan

**Target:** Reduce end-to-end latency by 150-250ms through buffer optimization and Opus codec enablement

**Estimated Time:** 2-4 hours for all phases

**Risk Level:** Low to Medium (test each phase thoroughly)

---

## Current State Analysis

### Measured Latency Issues

1. **JACK Input Buffer:** 32ms (256 frames @ 8kHz)
2. **Audio Chunk Accumulation:** 100ms (chunk_duration_s=0.1)
3. **AGC Window:** 1.5 seconds processing window
4. **Unbounded Output Queue:** Can accumulate 100-500ms
5. **Sample Rate Mismatch:** 8kHz JACK → 24kHz OpenAI requires resampling (5-10ms overhead)
6. **Codec:** G.711 ulaw @ 8kHz (telephony quality)

### Total Current Latency (estimated)
- **Input path:** ~150-200ms
- **Output path:** ~150-200ms
- **Total buffering overhead:** ~300-400ms (excludes network and processing)

---

## Phase 1: Quick Wins (30 minutes, ~150ms savings)

### 1.1 Reduce Audio Chunk Size

**File:** `/xfiles/update_plugins/mr_sip/src/mr_sip/sip_client_s2s.py`

**Line ~95-100:**
```python
# BEFORE:
self.audio_capture = JACKAudioCapture(
    target_sample_rate=24000,
    chunk_duration_s=0.1,  # ← CHANGE THIS
    chunk_callback=self._on_audio_chunk_from_jack,
    stereo_mix=True,
    agc_target_rms=0.15,
    agc_max_gain=20.0
)

# AFTER:
self.audio_capture = JACKAudioCapture(
    target_sample_rate=24000,
    chunk_duration_s=0.020,  # ← 20ms chunks (was 100ms)
    chunk_callback=self._on_audio_chunk_from_jack,
    stereo_mix=True,
    agc_target_rms=0.0,  # ← Disable AGC (see 1.2)
    agc_max_gain=20.0
)
```

**Expected Savings:** 80ms on input path

---

### 1.2 Disable or Reduce AGC Window

**Option A: Disable AGC (Recommended for phone calls)**

Already done in 1.1 above by setting `agc_target_rms=0.0`

**Option B: Reduce AGC Window (if AGC needed)**

**File:** `/xfiles/update_plugins/mr_sip/src/mr_sip/audio/jack_input_capture.py`

**Line ~60:**
```python
# BEFORE:
self.agc = SlidingWindowAGC(
    target_rms=self.agc_target_rms,
    max_gain=self.agc_max_gain,
    sample_rate=self.server_rate,
    window_seconds=1.5,  # ← CHANGE THIS
    smoothing=0.95
)

# AFTER:
self.agc = SlidingWindowAGC(
    target_rms=self.agc_target_rms,
    max_gain=self.agc_max_gain,
    sample_rate=self.server_rate,
    window_seconds=0.1,  # ← 100ms window (was 1.5s)
    smoothing=0.95
)
```

**Expected Savings:** 50-100ms processing time

**Rationale:** Phone audio from Telnyx is already normalized; AGC adds latency without much benefit.

---

### 1.3 Add Output Queue Limit

**File:** `/xfiles/update_plugins/mr_sip/src/mr_sip/sip_manager.py`

**Line ~25:**
```python
# BEFORE:
def __init__(self, log_id: str, destination: str, baresip_bot=None):
    self.log_id = log_id
    self.destination = destination
    self.baresip_bot = baresip_bot
    self.created_at = datetime.now()
    self.is_active = False
    self.halt_audio_out = False
    self.audio_queue = asyncio.Queue()  # ← CHANGE THIS

# AFTER:
def __init__(self, log_id: str, destination: str, baresip_bot=None):
    self.log_id = log_id
    self.destination = destination
    self.baresip_bot = baresip_bot
    self.created_at = datetime.now()
    self.is_active = False
    self.halt_audio_out = False
    self.audio_queue = asyncio.Queue(maxsize=10)  # ← Limit to ~200ms of audio
```

**Line ~85 (add error handling):**
```python
# BEFORE:
async def send_audio(self, audio_chunk: bytes):
    if self.is_active:
        self._audio_queued_count += 1
        try:
            await self.audio_queue.put(audio_chunk)

# AFTER:
async def send_audio(self, audio_chunk: bytes):
    if self.is_active:
        self._audio_queued_count += 1
        try:
            # Non-blocking put with timeout to prevent queue buildup
            await asyncio.wait_for(
                self.audio_queue.put(audio_chunk),
                timeout=0.1
            )
        except asyncio.TimeoutError:
            # Queue full - drop this chunk to prevent latency accumulation
            logger.warning(f"Audio queue full for session {self.log_id}, dropping chunk")
            return
```

**Expected Savings:** Prevents accumulation of 100-500ms latency spikes

---

### Phase 1 Testing

```bash
# 1. Make changes above
# 2. Restart your application
# 3. Make a test call
# 4. Listen for:
#    - Improved responsiveness (should feel snappier)
#    - No audio glitches or dropouts
#    - Check logs for "Audio queue full" warnings (should be rare)
```

**Success Criteria:**
- Noticeable improvement in response time
- No audio quality degradation
- Minimal queue full warnings (<1% of chunks)

---

## Phase 2: JACK Optimization (1 hour, ~30-50ms savings)

### 2.1 Reduce JACK Buffer Size

**File:** `/xfiles/update_plugins/mr_sip/src/mr_sip/start_jack_daemon.sh`

**Line ~10-12:**
```bash
# BEFORE:
SAMPLE_RATE=8000
PERIOD_SIZE=256        # 32ms at 8000 Hz
WAIT_TIME=32000

# AFTER (Conservative):
SAMPLE_RATE=8000
PERIOD_SIZE=128        # 16ms at 8000 Hz
WAIT_TIME=16000

# OR (Aggressive - test stability):
SAMPLE_RATE=8000
PERIOD_SIZE=64         # 8ms at 8000 Hz
WAIT_TIME=8000
```

**Expected Savings:** 16-24ms on both input and output paths (32-48ms total)

---

### 2.2 Test for XRUNs (Audio Glitches)

```bash
# 1. Stop existing JACK
killall jackd

# 2. Start with new settings
cd /xfiles/update_plugins/mr_sip/src/mr_sip
./start_jack_daemon.sh

# 3. Monitor for xruns
tail -f /tmp/jackd.log | grep -i xrun

# 4. Make test calls and listen for audio glitches
```

**If you see xruns or audio glitches:**
- Increase PERIOD_SIZE back to 128 or 256
- Check CPU usage during calls
- Consider keeping 256 if system is CPU-constrained

---

### Phase 2 Testing

**Success Criteria:**
- No xruns in JACK log during 5-minute test call
- No audio glitches or dropouts
- CPU usage remains reasonable (<80%)
- Further improvement in latency

---

## Phase 3: Opus Codec Enablement (1-2 hours, ~40-60ms savings)

### 3.1 Check Current Baresip Config

**File:** `~/.baresipy/config` (note: baresipy, not baresip)

```bash
# View current config
cat ~/.baresipy/config | grep -A 5 "audio_codecs\|opus"
```

Look for:
```ini
audio_codecs           PCMU/8000/1
audio_codecs           PCMA/8000/1
# opus_bitrate         28000  # ← May already be present
```

---

### 3.2 Enable Opus in Baresip Config

**File:** `~/.baresipy/config`

**Add/modify these lines:**
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

**Note:** If using older baresip, opus/24000/1 might not be supported. Try:
```ini
audio_codecs           opus/48000/2
```

---

### 3.3 Update JACK to 24kHz (Critical for Opus Benefits)

**File:** `/xfiles/update_plugins/mr_sip/src/mr_sip/start_jack_daemon.sh`

**Line ~10:**
```bash
# BEFORE:
SAMPLE_RATE=8000
PERIOD_SIZE=128        # From Phase 2
WAIT_TIME=16000

# AFTER:
SAMPLE_RATE=24000      # ← Match OpenAI and Opus!
PERIOD_SIZE=128        # 5.3ms at 24kHz (was 16ms at 8kHz)
WAIT_TIME=5333         # Match period time in microseconds
```

**Update the echo statements:**
```bash
echo "Period Size: ${PERIOD_SIZE} frames (${PERIOD_SIZE}/${SAMPLE_RATE} = $(echo "scale=1; ${PERIOD_SIZE}*1000/${SAMPLE_RATE}" | bc)ms)"
```

**Expected Savings:** 
- Eliminates resampling overhead: 10-20ms
- Smaller JACK buffer at 24kHz: additional 10-15ms
- Total: 20-35ms

---

### 3.4 Verify Opus Module in Baresip

**Option A: Check if opus module exists**
```bash
# Look for opus module
ls ~/.baresipy/modules/ | grep opus
# or
ls /usr/lib/baresip/modules/ | grep opus
```

**Option B: Try loading in code**

**File:** `/xfiles/update_plugins/mr_sip/src/mr_sip/audio_handler.py`

**Line ~30 (in configure_baresip_jack method):**
```python
def configure_baresip_jack(self, baresip_bot):
    if baresip_bot:
        try:
            # Try to load opus module
            try:
                baresip_bot.do_command("/module_load opus")
                logger.info("Loaded Opus codec module")
            except Exception as e:
                logger.warning(f"Could not load Opus module: {e}")
            
            # Rest of existing code...
            if os.environ.get("BARESIP_JACK_V", "0") == "1":
                baresip_bot.do_command("/module_load jack")
                baresip_bot.do_command("/ausrc jack,MindRootSIP.*")
                baresip_bot.do_command("/auplay jack,MR-STT")
```

---

### 3.5 Configure Telnyx for Opus

**Via Telnyx Portal:**

1. Log into Telnyx portal
2. Navigate to: **Voice → SIP Connections → [Your Connection]**
3. Find **Codecs** or **Audio Settings** section
4. Enable and prioritize:
   ```
   1. opus (highest priority)
   2. PCMU (G.711 μ-law fallback)
   3. PCMA (G.711 A-law fallback)
   ```
5. Save settings

**Note:** Look for "HD Voice" or "Wideband" settings - enabling these typically enables Opus.

---

### 3.6 Verify Opus is Active During Call

**Add logging to verify codec:**

**File:** `/xfiles/update_plugins/mr_sip/src/mr_sip/sip_client_s2s.py`

**Line ~50 (in handle_call_established):**
```python
def handle_call_established(self):
    logger.info("=== CALL ESTABLISHED (S2S Mode) ===")
    
    # Check active codec
    try:
        # This may not work with all baresip versions
        codec_info = self.do_command("/call_status")
        logger.info(f"Active codec: {codec_info}")
    except:
        pass
    
    self.call_start_time = datetime.now()
    # ... rest of existing code
```

**Look for in logs:**
```
Active codec: opus/24000/1
# or
Active codec: opus/48000/2
```

If you see `PCMU/8000/1`, Opus negotiation failed.

---

### Phase 3 Testing

```bash
# 1. Update baresip config
# 2. Update JACK to 24kHz
# 3. Restart JACK
killall jackd
cd /xfiles/update_plugins/mr_sip/src/mr_sip
./start_jack_daemon.sh

# 4. Verify JACK rate
jack_samplerate
# Should show: 24000

# 5. Make test call
# 6. Check logs for:
#    - "Active codec: opus"
#    - No resampling messages
#    - Improved audio quality
```

**Success Criteria:**
- Opus codec active (check logs)
- JACK running at 24kHz
- No audio glitches
- Noticeably better audio quality
- Further latency improvement

**Fallback Plan:**
If Opus causes issues:
1. Remove opus from baresip config
2. Keep JACK at 24kHz (still benefits from less resampling)
3. Or revert JACK to 8kHz if 24kHz causes problems

---

## Phase 4: VPS Migration (2-4 hours, ~240ms savings)

### 4.1 Deploy to Virginia VPS

**Recommended Providers:**
- AWS EC2 (us-east-1)
- DigitalOcean (NYC3 or NYC1)
- Linode (Newark)
- Vultr (New Jersey)

**Minimum Specs:**
- 2 vCPU
- 4GB RAM
- Ubuntu 22.04 LTS
- Low-latency network (not shared hosting)

---

### 4.2 Installation on VPS

```bash
# 1. Install dependencies
sudo apt update
sudo apt install -y python3.11 python3-pip jackd2 libjack-jackd2-dev

# 2. Clone/copy your code
# (your deployment process here)

# 3. Install Python packages
cd /path/to/mr_sip
pip install -r requirements.txt

# 4. Configure JACK for headless operation
# Edit /etc/security/limits.conf:
@audio   -  rtprio     95
@audio   -  memlock    unlimited

# 5. Start JACK daemon
cd src/mr_sip
./start_jack_daemon.sh
```

---

### 4.3 Test Latency from VPS

```bash
# From VPS, ping both endpoints
ping -c 10 sip.telnyx.com
# Should see ~35ms (vs 73ms from McAllen)

ping -c 10 api.openai.com
# Should see ~18ms (vs 24ms from McAllen)
```

**Expected Savings:**
- Telnyx path: 73ms → 35ms = 38ms × 2 = 76ms
- OpenAI path: 24ms → 18ms = 6ms × 2 = 12ms
- Eliminates McAllen double-hop: ~146ms
- **Total: ~234ms savings**

---

## Verification & Monitoring

### Add Latency Tracking

**File:** `/xfiles/update_plugins/mr_sip/src/mr_sip/sip_client_s2s.py`

**Add timestamp tracking:**
```python
async def _on_audio_chunk_from_jack(self, audio_chunk: np.ndarray):
    try:
        if not hasattr(self, '_audio_chunk_count'):
            self._audio_chunk_count = 0
            self._chunk_timestamps = {}
        
        self._audio_chunk_count += 1
        chunk_id = self._audio_chunk_count
        
        # Track when chunk was captured
        import time
        self._chunk_timestamps[chunk_id] = time.time()
        
        # ... existing code to send to OpenAI ...
```

**In OpenAI response handler:**

**File:** `/xfiles/plugins_ah/ah_openai/src/ah_openai/speech_to_speech.py`

**Line ~80 (in on_message):**
```python
elif server_event['type'] == "response.output_audio.delta":
    audio_bytes = base64.b64decode(server_event['delta'])
    
    # Calculate latency (if we had chunk_id tracking)
    # This is simplified - you'd need to correlate chunks
    import time
    response_time = time.time()
    # logger.info(f"Audio response latency: {(response_time - input_time)*1000:.1f}ms")
    
    # ... existing code ...
```

---

### Monitor Key Metrics

**Add to logs:**
1. **JACK xruns:** `grep xrun /tmp/jackd.log`
2. **Queue full events:** `grep "Audio queue full" /path/to/logs`
3. **Codec in use:** Check call establishment logs
4. **CPU usage:** `top` or `htop` during calls
5. **Network latency:** Periodic pings to Telnyx and OpenAI

---

## Rollback Procedures

### If Phase 1 Causes Issues

```python
# Revert chunk size
chunk_duration_s=0.1

# Re-enable AGC
agc_target_rms=0.15

# Remove queue limit
self.audio_queue = asyncio.Queue()
```

### If Phase 2 Causes Issues

```bash
# Revert JACK buffer
PERIOD_SIZE=256
WAIT_TIME=32000
```

### If Phase 3 Causes Issues

```ini
# Remove opus from baresip config
# Keep only:
audio_codecs           PCMU/8000/1
audio_codecs           PCMA/8000/1
```

```bash
# Revert JACK to 8kHz
SAMPLE_RATE=8000
PERIOD_SIZE=128  # or 256
```

---

## Expected Results Summary

| Phase | Time | Latency Savings | Risk | Reversible |
|-------|------|-----------------|------|------------|
| Phase 1: Quick Wins | 30 min | ~150ms | Low | Yes |
| Phase 2: JACK Optimization | 1 hour | ~30-50ms | Medium | Yes |
| Phase 3: Opus Enablement | 1-2 hours | ~40-60ms | Medium | Yes |
| Phase 4: VPS Migration | 2-4 hours | ~240ms | Medium | Yes |
| **TOTAL** | **4-7.5 hours** | **460-500ms** | - | - |

---

## Key Files Reference

### Configuration Files
- `~/.baresipy/config` - Baresip codec and audio settings
- `/xfiles/update_plugins/mr_sip/src/mr_sip/start_jack_daemon.sh` - JACK daemon config

### Code Files to Modify
- `/xfiles/update_plugins/mr_sip/src/mr_sip/sip_client_s2s.py` - Audio capture settings
- `/xfiles/update_plugins/mr_sip/src/mr_sip/audio/jack_input_capture.py` - AGC settings
- `/xfiles/update_plugins/mr_sip/src/mr_sip/sip_manager.py` - Output queue management
- `/xfiles/update_plugins/mr_sip/src/mr_sip/audio_handler.py` - Opus module loading

### Monitoring
- `/tmp/jackd.log` - JACK daemon logs (xruns, errors)
- Application logs - Check for codec info, queue warnings

---

## Notes for Implementation

1. **Test each phase independently** - Don't combine all changes at once
2. **Make one change at a time** within each phase
3. **Keep backups** of config files before modifying
4. **Test with real calls** after each change
5. **Monitor logs** for warnings and errors
6. **Measure latency** before and after each phase
7. **Document results** for each phase

---

## Success Criteria

### Phase 1 Success
- [ ] Chunk size reduced to 20ms
- [ ] AGC disabled or window reduced to 100ms
- [ ] Output queue limited to 10 items
- [ ] No audio quality degradation
- [ ] Noticeable latency improvement

### Phase 2 Success
- [ ] JACK buffer reduced to 128 or 64 frames
- [ ] No xruns during 5-minute test call
- [ ] No audio glitches
- [ ] Further latency improvement

### Phase 3 Success
- [ ] Opus codec active during calls
- [ ] JACK running at 24kHz
- [ ] No resampling overhead
- [ ] Better audio quality
- [ ] Further latency improvement

### Phase 4 Success
- [ ] Application running on Virginia VPS
- [ ] Ping times: ~35ms to Telnyx, ~18ms to OpenAI
- [ ] All previous optimizations working
- [ ] Total latency reduced by 400-500ms

---

## Contact & Support

If issues arise during implementation:
1. Check logs first (`/tmp/jackd.log`, application logs)
2. Verify each change was applied correctly
3. Test rollback procedures
4. Document the issue with logs and symptoms

**Good luck with the implementation!**
