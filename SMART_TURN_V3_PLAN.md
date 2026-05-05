Smart Turn v3 Integration Plan for mr_sip

Overview: Replace Silero VAD with Pipecat Smart Turn v3 for end-of-turn detection.
Smart Turn v3 is an 8MB ONNX model that runs in about 12ms on CPU and determines
whether a user has finished their conversational turn based on audio prosody.

## Current Architecture (Silero VAD + Cohere Transcribe)

SIP ulaw 8kHz frames (160B/20ms) flow through Silero VAD with dual-threshold
detection (start=0.5, end=0.3). Speech start fires barge-in callback. Speech end
at eager_silence_ms=500ms triggers transcription and emits eager EOT. A confirmation
timer at final_silence_ms=700ms emits final EOT. Cohere Transcribe runs on remote
HTTP at 16kHz.

Key files:
- src/mr_sip/stt/silero_cohere_stt.py - Main STT provider with VAD
- src/mr_sip/stt/stt_factory.py - Provider factory
- src/mr_sip/stt/base_stt.py - Base STT interface
- src/mr_sip/sip_client_s2s.py - SIP client (feeds audio to STT)
- src/mr_sip/sip_manager.py - Session/audio queue management

## Proposed Architecture (Smart Turn v3)

SIP ulaw 8kHz frames (160B/20ms) -> Simple RMS energy detector for speech start/ongoing detection. Speech start fires barge-in callback. During speech, buffer audio and resample to 16kHz. Smart Turn v3 polling loop runs every 80ms, feeding accumulated 16kHz speech buffer (up to 8s) to ONNX model. Model outputs turn_complete probability (0-1). When prob > 0.5 AND silence detected for N ms, trigger transcription. Cohere Transcribe (remote HTTP, 16kHz) remains unchanged, then text to agent.

## Key Design Decisions

### 1. Speech Detection: Simple RMS Energy Detector

Smart Turn v3 is designed to work WITH a VAD, not replace speech detection. We need a lightweight way to know when the user is speaking vs silent. Replace Silero's neural VAD with a simple RMS energy threshold: compute RMS on each 20ms ulaw frame; if RMS > threshold for N consecutive frames -> speech start; if RMS < threshold for M consecutive frames -> potential silence. This is extremely cheap (no ML inference) and runs inline with audio receipt.

Parameters:
- speech_start_rms: RMS threshold for speech (default: 200)
- speech_start_frames: Consecutive frames above threshold to trigger start (default: 3 = 60ms)
- silence_frames_for_turn_check: Frames below threshold before running Smart Turn (default: 10 = 200ms)

### 2. Smart Turn v3 Polling at 80ms

Instead of VAD-triggered turn detection, we poll Smart Turn every 80ms during speech and for a short window after silence is detected. Smart Turn inference takes ~12ms on CPU, ~3-6ms on GPU. 80ms gives plenty of headroom while being responsive. At 80ms polling, worst-case added latency is 80ms (acceptable). On H200, we could even do 40ms or 20ms, but 80ms is a good balance.

Polling logic:
1. Start polling when speech is first detected
2. Continue polling while user is speaking (every 80ms)
3. When silence is detected, continue polling for up to max_silence_poll_ms (e.g., 2000ms)
4. If Smart Turn returns prob > 0.5, trigger transcription immediately
5. If max silence poll time expires without Smart Turn trigger, fall back to transcription anyway (safety net)

### 3. Audio Buffer Management

Smart Turn v3 expects 16kHz mono PCM float32 audio, up to 8 seconds. Maintain a rolling buffer of the last 8 seconds of speech audio at 16kHz. On each poll: resample accumulated ulaw buffer to 16kHz, truncate to last 8s, pad with silence at beginning if shorter than 8s. Resampling: ulaw 8kHz -> PCM int16 -> float32 -> linear interpolation 2x -> 16kHz float32 (already implemented in SileroCohereSTT._resample_2x()).

### 4. Simplified Turn Detection (No Eager/Final Two-Stage)

Smart Turn v3's output is a direct turn-complete probability. This simplifies the two-stage eager/final EOT system. Old: VAD detects silence -> transcribe (eager) -> wait confirmation -> final. New: Smart Turn says turn complete -> transcribe -> emit as final immediately. This eliminates the eager/final complexity and the confirmation timer. The tradeoff is we lose the early preparation benefit of eager EOT, but Smart Turn's accuracy (~94% for English) should make this worthwhile. Optional hybrid: keep a short confirmation window (200-300ms) after Smart Turn triggers to catch false positives.

### 5. ONNX Model Loading

Smart Turn v3 is distributed as an ONNX model (~8MB int8 quantized). Dependencies: onnxruntime, transformers (for WhisperFeatureExtractor), numpy (all already available). Model file: download from HuggingFace pipecat-ai/smart-turn-v3 (smart-turn-v3.1.onnx or latest). Bundle with mr_sip or download on first use. Path configurable via SMART_TURN_MODEL_PATH env var. Inference code from pipecat-ai/smart-turn inference.py uses WhisperFeatureExtractor with chunk_length=8, ONNX session with CPU execution provider, and predict_endpoint() returning prediction (1/0) and probability.

## Implementation Plan

### Step 1: Create smart_turn_v3_stt.py

New file at src/mr_sip/stt/smart_turn_v3_stt.py. This is a new STT provider that extends BaseSTTProvider (same interface as SileroCohereSTT). It uses RMS energy detection for speech start/end, runs Smart Turn v3 ONNX model on a polling loop, triggers Cohere Transcribe when turn is detected, and emits results via the same callback interface.

Class SmartTurnV3STT key methods:
- start(): Load ONNX model, initialize feature extractor, start polling loop
- stop(): Stop polling, release resources
- add_audio_bytes(ulaw_bytes): Feed audio, update RMS detector, buffer speech
- _poll_smart_turn(): Async loop running every 80ms
- _run_smart_turn_inference(audio_float16k): Run ONNX inference in executor
- _on_speech_start(): Fire barge-in callback
- _on_turn_complete(): Trigger transcription, emit result

Configuration env vars:
- SMART_TURN_MODEL_PATH: Path to ONNX model file
- SMART_TURN_POLL_MS: Polling interval (default: 80)
- SMART_TURN_THRESHOLD: Probability threshold for turn complete (default: 0.5)
- SMART_TURN_MAX_SILENCE_POLL_MS: Max silence before fallback (default: 2000)
- SMART_TURN_SPEECH_START_RMS: RMS threshold for speech (default: 200)
- SMART_TURN_SPEECH_START_FRAMES: Consecutive frames for speech start (default: 3)
- SMART_TURN_SILENCE_FRAMES: Silence frames before turn check (default: 10)
- SMART_TURN_CONFIRMATION_MS: Optional confirmation window (default: 0 = disabled)

### Step 2: Update stt_factory.py

Add smart_turn_v3 as a new provider option. When provider_name is smart_turn_v3, import SmartTurnV3STT from .smart_turn_v3_stt, pop api_key and encoding kwargs, and return SmartTurnV3STT(**kwargs).

### Step 3: Download/Bundle ONNX Model

Download smart-turn-v3.1.onnx from HuggingFace. Place in src/mr_sip/models/ or a configurable path. Add to .gitignore or use Git LFS. Add download script at scripts/download_smart_turn_model.sh.

### Step 4: Testing

1. Unit test RMS detector with synthetic audio
2. Unit test Smart Turn inference with sample audio files
3. Integration test with recorded SIP audio
4. Live test call with STT_PROVIDER=smart_turn_v3

### Step 5: Performance Tuning

Measure end-to-end latency: speech end -> Smart Turn trigger -> transcription -> text. Compare against current Silero VAD baseline. Tune polling interval, RMS thresholds, confirmation window. Consider GPU inference via onnxruntime-gpu for even lower latency.

## Migration Path

1. Implement SmartTurnV3STT alongside existing SileroCohereSTT
2. Both use the same BaseSTTProvider interface, making it a drop-in replacement
3. Switch via STT_PROVIDER=smart_turn_v3 env var
4. A/B test by running some calls with each provider
5. Once validated, Silero VAD dependency can be removed

## Expected Benefits

1. Better turn detection accuracy: Smart Turn v3 uses linguistic/prosodic cues, not just audio energy. About 94% accuracy for English vs Silero's energy-only approach.
2. Fewer false turn-ends: Smart Turn can distinguish pauses from actual turn completion, reducing premature interruptions.
3. Simpler code: Eliminates dual-threshold VAD, eager/final two-stage EOT, and confirmation timer complexity.
4. CPU-friendly: 8MB model, about 12ms inference on CPU. On H200 GPU even faster.
5. Multi-language: Supports 23 languages out of the box.

## Potential Risks and Mitigations

1. Smart Turn not designed for polling: The model expects to run when VAD detects silence, not continuously. Mitigation: Only poll during speech plus short silence window, not continuously.
2. Increased CPU usage: Polling every 80ms adds about 12ms of CPU work per poll. Mitigation: On H200 this is negligible. Can reduce poll frequency if needed.
3. WhisperFeatureExtractor overhead: The feature extractor may add latency. Mitigation: Profile and optimize. Consider caching feature extractor outputs for overlapping audio windows.
4. Model accuracy for short utterances: Smart Turn works best with more context. Mitigation: Keep the fallback silence timeout for very short utterances.

## Open Questions

1. Should we keep a short confirmation window (200-300ms) after Smart Turn triggers to reduce false positives? This would add latency but improve accuracy.
2. Should we run Smart Turn on GPU via onnxruntime-gpu? The H200 has plenty of GPU memory. This would reduce inference time to about 3-4ms.
3. Should we keep the eager/final two-stage system? Smart Turn could trigger eager transcription with a short confirmation for final, preserving the latency benefit of eager EOT.
4. Model file management: Bundle with the plugin, download on first use, or require pre-installation?
