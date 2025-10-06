# MindRoot SIP Plugin - TODO & Refactoring Plan

## Remaining Implementation Tasks

### Critical Issues to Fix

1. **Audio Output Implementation**
   - [ ] Implement actual audio injection into baresip calls in `_send_tts_audio()` method
   - [ ] Research baresip audio input APIs or alternative methods for TTS playback
   - [ ] Test audio format conversion between ElevenLabs output and baresip input
   - [ ] Handle audio synchronization and buffering issues

2. **Service Manager Integration**
   - [x] ~~Fix mr_eleven_stream to properly call sip_audio_out_chunk~~ (Not needed - context is auto-added)
   - [ ] Verify service_manager automatically adds context parameter
   - [ ] Test integration between mr_eleven_stream and mr_sip plugins

3. **Error Handling & Robustness**
   - [ ] Add comprehensive error handling for SIP connection failures
   - [ ] Implement retry logic for failed calls
   - [ ] Handle network disconnections gracefully
   - [ ] Add timeout handling for call establishment
   - [ ] Implement proper cleanup on unexpected failures

### Testing & Validation

4. **Integration Testing**
   - [ ] Test with actual SIP provider (VoIP.ms)
   - [ ] Verify audio quality and latency
   - [ ] Test conversation flow: speech → transcription → agent → TTS → audio
   - [ ] Test multiple concurrent calls (if supported)
   - [ ] Test call termination and cleanup

5. **Audio Pipeline Testing**
   - [ ] Verify Whisper VAD sensitivity settings for phone audio
   - [ ] Test audio format conversions (8kHz ↔ 16kHz)
   - [ ] Validate TTS audio quality over SIP
   - [ ] Test with different phone line qualities

### Documentation & Configuration

6. **Configuration Management**
   - [ ] Add configuration validation on startup
   - [ ] Create configuration templates for different SIP providers
   - [ ] Add environment variable documentation
   - [ ] Implement configuration hot-reloading

7. **Logging & Monitoring**
   - [ ] Add structured logging for call events
   - [ ] Implement call metrics and statistics
   - [ ] Add debug modes for troubleshooting
   - [ ] Create call session logging

### Advanced Features

8. **Enhanced Functionality**
   - [ ] Support for incoming call handling
   - [ ] Multiple simultaneous call support
   - [ ] Call recording and playback
   - [ ] Call transfer capabilities
   - [ ] DTMF tone handling

9. **Performance Optimization**
   - [ ] Optimize audio buffer sizes
   - [ ] Implement adaptive audio quality
   - [ ] Add connection pooling for SIP sessions
   - [ ] Optimize memory usage for long calls

## Refactoring Plan

### Current File Structure Issues

**Problems:**
- `mod.py` is too large (366 lines) with mixed responsibilities
- `baresip_integration.py` is very large (474 lines) and complex
- No clear separation between commands, services, and core logic
- Difficult to test individual components

### Proposed Refactored Structure

```
mr_sip/
├── src/
│   └── mr_sip/
│       ├── __init__.py
│       ├── commands.py          # User-facing commands (call, hangup)
│       ├── services.py          # Internal services (dial_service, sip_audio_out_chunk, etc.)
│       ├── core/
│       │   ├── __init__.py
│       │   ├── sip_client.py    # SIP client wrapper (from baresip_integration.py)
│       │   ├── session_manager.py  # Session management (from sip_manager.py)
│       │   ├── audio_processor.py  # Audio processing and VAD
│       │   └── transcription.py    # Whisper integration
│       ├── config/
│       │   ├── __init__.py
│       │   ├── settings.py      # Configuration management
│       │   └── validation.py    # Config validation
│       └── utils/
│           ├── __init__.py
│           ├── audio_utils.py   # Audio format conversion utilities
│           └── logging_utils.py # Logging helpers
├── tests/
│   ├── test_commands.py
│   ├── test_services.py
│   ├── test_sip_client.py
│   └── test_audio_processing.py
├── plugin_info.json
├── pyproject.toml
├── setup.py
└── README.md
```

### Refactoring Steps

#### Phase 1: Extract Commands and Services

1. **Create `commands.py`**
   ```python
   # Extract from mod.py:
   # - call() command
   # - hangup() command
   # - Associated helper functions
   ```

2. **Create `services.py`**
   ```python
   # Extract from mod.py:
   # - dial_service()
   # - sip_audio_out_chunk()
   # - end_call_service()
   # - quit() hook
   ```

3. **Update `mod.py`**
   ```python
   # Keep only:
   # - Plugin initialization
   # - Import statements for commands and services
   # - Global configuration
   ```

#### Phase 2: Break Down Core Components

4. **Create `core/sip_client.py`**
   ```python
   # Extract from baresip_integration.py:
   # - MindRootSIPBot class
   # - Core SIP functionality
   # - Call handling logic
   ```

5. **Create `core/audio_processor.py`**
   ```python
   # Extract from baresip_integration.py:
   # - Audio processing functions
   # - Format conversion
   # - Real-time audio handling
   ```

6. **Create `core/transcription.py`**
   ```python
   # Extract from baresip_integration.py:
   # - Whisper VAD integration
   # - Transcription callbacks
   # - Utterance processing
   ```

7. **Refactor `core/session_manager.py`**
   ```python
   # Simplify existing sip_manager.py:
   # - Remove complex audio handling (move to audio_processor)
   # - Focus on session lifecycle
   # - Improve async patterns
   ```

#### Phase 3: Configuration and Utilities

8. **Create `config/settings.py`**
   ```python
   # Extract configuration logic:
   # - Environment variable handling
   # - Default values
   # - Configuration classes
   ```

9. **Create `utils/audio_utils.py`**
   ```python
   # Extract audio utilities:
   # - Format conversion functions
   # - Audio validation
   # - Buffer management
   ```

10. **Create `utils/logging_utils.py`**
    ```python
    # Standardize logging:
    # - Structured logging setup
    # - Call event logging
    # - Debug helpers
    ```

### Benefits of Refactoring

1. **Maintainability**
   - Smaller, focused files
   - Clear separation of concerns
   - Easier to understand and modify

2. **Testability**
   - Individual components can be unit tested
   - Mock dependencies more easily
   - Better test coverage

3. **Reusability**
   - Core components can be reused
   - Easier to extend functionality
   - Plugin architecture more modular

4. **Development**
   - Multiple developers can work on different components
   - Reduced merge conflicts
   - Clearer code ownership

### Migration Strategy

1. **Backward Compatibility**
   - Keep existing `mod.py` importing from new modules
   - Maintain same external API
   - No changes to plugin_info.json

2. **Incremental Migration**
   - Refactor one component at a time
   - Test after each migration step
   - Keep git history clean with focused commits

3. **Testing Strategy**
   - Add tests for each extracted component
   - Maintain integration tests
   - Test plugin loading and registration

## Handoff Notes for Engineer

### Priority Order

1. **HIGH**: Fix audio output implementation (critical for functionality)
2. **HIGH**: Complete integration testing with real SIP calls
3. **MEDIUM**: Implement refactoring plan (improves maintainability)
4. **MEDIUM**: Add comprehensive error handling
5. **LOW**: Advanced features and optimizations

### Key Technical Challenges

1. **Audio Injection**: The biggest technical challenge is getting TTS audio into the baresip call. This may require:
   - Custom baresip module development
   - Alternative audio routing approaches
   - Real-time audio mixing

2. **Timing and Synchronization**: Voice conversations require careful timing:
   - VAD sensitivity tuning
   - Audio buffer management
   - Preventing audio feedback loops

3. **Format Compatibility**: Multiple audio format conversions:
   - SIP (μ-law 8kHz) ↔ Whisper (PCM 16kHz) ↔ ElevenLabs (various formats)

### Resources and References

- **Baresip Documentation**: https://github.com/baresip/baresip
- **Original Code**: `/files/whispertest/baresip_transcriber.py`
- **MindRoot Architecture**: `/files/mindroot/src/mindroot/lib/providers/`
- **ElevenLabs Integration**: `/xfiles/upd5/mr_eleven_stream/`

### Testing Environment Setup

1. Install system dependencies: `libsndfile1-dev`
2. Configure baresip with sndfile module
3. Set up SIP account credentials
4. Configure ElevenLabs API key
5. Test with actual phone calls to validate end-to-end functionality

This refactoring plan will make the codebase much more maintainable and testable while preserving all existing functionality.
