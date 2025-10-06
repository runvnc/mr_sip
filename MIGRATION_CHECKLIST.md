# Migration Checklist: Deepgram Integration

## Pre-Migration

- [ ] Review current system performance and latency
- [ ] Document current configuration
- [ ] Backup current codebase
- [ ] Test current system to establish baseline

## Installation

- [ ] Install new dependencies:
  ```bash
  pip install inotify websockets
  ```

- [ ] Verify installations:
  ```bash
  python -c "import inotify; print('inotify OK')"
  python -c "import websockets; print('websockets OK')"
  ```

## Configuration

### For Deepgram

- [ ] Sign up for Deepgram account at https://deepgram.com/
- [ ] Create API key
- [ ] Set environment variable:
  ```bash
  export DEEPGRAM_API_KEY="your_key_here"
  export STT_PROVIDER="deepgram"
  ```
- [ ] Verify API key works:
  ```bash
  curl -H "Authorization: Token $DEEPGRAM_API_KEY" \
       https://api.deepgram.com/v1/projects
  ```

### For Whisper (Alternative)

- [ ] Set environment variables:
  ```bash
  export STT_PROVIDER="whisper_vad"
  export STT_MODEL_SIZE="small"
  ```

## Code Updates

- [ ] Update imports in services.py (if needed)
- [ ] Change `dial_service` calls to `dial_service_v2`
- [ ] Update any direct references to `MindRootSIPBot` to `MindRootSIPBotV2`
- [ ] Review and update error handling for new STT providers

## Testing

### Unit Tests

- [ ] Test STT factory:
  ```python
  from mr_sip.stt import create_stt_provider
  stt = create_stt_provider('deepgram', api_key='test')
  assert stt is not None
  ```

- [ ] Test inotify capture (with test file)
- [ ] Test Deepgram connection (with real API key)

### Integration Tests

- [ ] Make test call with Deepgram
- [ ] Verify audio is transcribed
- [ ] Check transcriptions appear in chat
- [ ] Measure latency (should be <500ms for finals)
- [ ] Test with Whisper fallback
- [ ] Test call hangup and cleanup

### Performance Tests

- [ ] Measure first transcription latency
- [ ] Measure average transcription latency
- [ ] Check CPU usage during call
- [ ] Monitor memory usage
- [ ] Test with multiple simultaneous calls (if applicable)

## Monitoring

- [ ] Set up logging for STT provider
- [ ] Monitor Deepgram API usage/costs
- [ ] Track transcription accuracy
- [ ] Monitor error rates
- [ ] Set up alerts for failures

## Rollback Plan

- [ ] Document rollback procedure:
  1. Revert to original `dial_service`
  2. Remove new environment variables
  3. Restart services

- [ ] Keep old code available:
  - `sip_client.py` (original)
  - `services.py` (original)

- [ ] Test rollback procedure

## Documentation

- [ ] Update README.md with new configuration
- [ ] Document environment variables
- [ ] Add troubleshooting guide
- [ ] Update API documentation
- [ ] Create user guide for Deepgram setup

## Production Deployment

- [ ] Deploy to staging environment first
- [ ] Run full test suite in staging
- [ ] Monitor staging for 24-48 hours
- [ ] Get approval for production deployment
- [ ] Deploy to production during low-traffic period
- [ ] Monitor production closely for first 24 hours

## Post-Deployment

- [ ] Verify all calls are working
- [ ] Check transcription quality
- [ ] Monitor latency metrics
- [ ] Review error logs
- [ ] Collect user feedback
- [ ] Document any issues and resolutions

## Cost Monitoring (Deepgram)

- [ ] Set up billing alerts
- [ ] Monitor daily API usage
- [ ] Calculate cost per call
- [ ] Compare with budget
- [ ] Optimize if needed (model selection, etc.)

## Future Enhancements

- [ ] Plan JACK audio capture implementation
- [ ] Evaluate streaming Whisper
- [ ] Consider additional STT providers
- [ ] Optimize for specific use cases

## Sign-off

- [ ] Technical lead approval
- [ ] QA sign-off
- [ ] Product owner approval
- [ ] Documentation complete
- [ ] Training completed (if needed)

---

## Notes

Date: _____________

Deployed by: _____________

Issues encountered:



Resolutions:



Performance improvements observed:



