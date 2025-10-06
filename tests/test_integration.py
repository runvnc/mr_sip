#!/usr/bin/env python3
"""
Integration test for MindRoot SIP plugin with JACK audio.

This test verifies that all components can be imported and basic
functionality works without requiring an actual SIP call.
"""

import asyncio
import logging
import numpy as np
import sys
import os

# Add the plugin to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

logger = logging.getLogger(__name__)

class MockContext:
    """Mock MindRoot context for testing."""
    def __init__(self, log_id):
        self.log_id = log_id

async def test_imports():
    """Test that all components can be imported."""
    print("Testing imports...")
    
    try:
        from mr_sip.jack_streamer import JACKAudioStreamer
        print("✓ JACKAudioStreamer imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import JACKAudioStreamer: {e}")
        return False
    
    try:
        from mr_sip.audio_handler import AudioHandler
        print("✓ AudioHandler imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import AudioHandler: {e}")
        return False
    
    try:
        from mr_sip.sip_client import MindRootSIPBot
        print("✓ MindRootSIPBot imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import MindRootSIPBot: {e}")
        return False
    
    try:
        from mr_sip.sip_manager import get_session_manager
        print("✓ SIP session manager imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import SIP session manager: {e}")
        return False
    
    try:
        import mr_sip.commands
        print("✓ Commands module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import commands: {e}")
        return False
    
    try:
        import mr_sip.services
        print("✓ Services module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import services: {e}")
        return False
    
    return True

async def test_audio_handler():
    """Test AudioHandler without JACK server."""
    print("\nTesting AudioHandler...")
    
    try:
        from mr_sip.audio_handler import AudioHandler
        
        handler = AudioHandler()
        print("✓ AudioHandler created successfully")
        
        # Test audio processing without JACK
        test_audio = np.random.randint(-32768, 32767, 1000, dtype=np.int16)
        test_bytes = test_audio.tobytes()
        
        # This should not crash even without JACK
        await handler.send_tts_audio(test_bytes)
        print("✓ Audio processing test completed (JACK not required)")
        
        return True
        
    except Exception as e:
        print(f"✗ AudioHandler test failed: {e}")
        return False

async def test_session_manager():
    """Test SIP session manager."""
    print("\nTesting SIP session manager...")
    
    try:
        from mr_sip.sip_manager import get_session_manager
        
        session_manager = get_session_manager()
        print("✓ Session manager created successfully")
        
        # Test session creation (without actual SIP bot)
        context = MockContext("test_session_123")
        session = await session_manager.create_session(
            log_id=context.log_id,
            destination="test_destination",
            baresip_bot=None
        )
        
        print(f"✓ Session created: {session.log_id}")
        
        # Test session retrieval
        retrieved_session = await session_manager.get_session(context.log_id)
        assert retrieved_session is not None
        print("✓ Session retrieval works")
        
        # Test session cleanup
        success = await session_manager.end_session(context.log_id)
        assert success
        print("✓ Session cleanup works")
        
        return True
        
    except Exception as e:
        print(f"✗ Session manager test failed: {e}")
        return False

async def test_jack_integration():
    """Test JACK integration if JACK server is available."""
    print("\nTesting JACK integration...")
    
    try:
        import jack
        
        # Try to create a JACK client
        try:
            client = jack.Client("TestClient")
            client.activate()
            print(f"✓ JACK client created (sample rate: {client.samplerate} Hz)")
            client.deactivate()
            
            # Test our JACK streamer
            from mr_sip.jack_streamer import JACKAudioStreamer
            
            streamer = JACKAudioStreamer("TestStreamer")
            streamer.start()
            print("✓ JACK streamer created and started")
            
            # Test audio writing
            test_audio = np.random.random(1000).astype(np.float32)
            streamer.write_audio(test_audio)
            print("✓ Audio writing to JACK buffer works")
            
            streamer.stop()
            print("✓ JACK streamer stopped successfully")
            
            return True
            
        except jack.JackError as e:
            print(f"⚠️ JACK server not available: {e}")
            print("  This is OK - JACK tests skipped")
            return True  # Not a failure, just not available
            
    except ImportError as e:
        print(f"⚠️ JACK-Client not available: {e}")
        print("  Install with: pip install JACK-Client")
        return False
    except Exception as e:
        print(f"✗ JACK integration test failed: {e}")
        return False

async def test_audio_format_conversion():
    """Test audio format conversion functions."""
    print("\nTesting audio format conversion...")
    
    try:
        from scipy import signal
        
        # Test resampling (simulating ElevenLabs 22050 Hz to telephony 8000 Hz)
        original_rate = 22050
        target_rate = 8000
        duration = 1.0  # 1 second
        
        # Generate test audio
        t = np.linspace(0, duration, int(original_rate * duration))
        test_audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)  # 440 Hz tone
        
        # Resample
        target_length = int(len(test_audio) * target_rate / original_rate)
        resampled = signal.resample(test_audio, target_length)
        
        print(f"✓ Resampled audio from {len(test_audio)} to {len(resampled)} samples")
        print(f"✓ Resampling ratio: {original_rate} Hz → {target_rate} Hz")
        
        # Test format conversion (16-bit PCM to float32)
        pcm_audio = (test_audio * 32767).astype(np.int16)
        float_audio = pcm_audio.astype(np.float32) / 32768.0
        
        print("✓ PCM to float32 conversion works")
        
        return True
        
    except Exception as e:
        print(f"✗ Audio format conversion test failed: {e}")
        return False

async def main():
    """Run all integration tests."""
    print("MindRoot SIP Plugin Integration Tests")
    print("====================================\n")
    
    # Setup logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise
    
    tests = [
        ("Component Imports", test_imports),
        ("Audio Handler", test_audio_handler),
        ("Session Manager", test_session_manager),
        ("JACK Integration", test_jack_integration),
        ("Audio Format Conversion", test_audio_format_conversion),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Running {test_name} test...")
        try:
            if await test_func():
                passed += 1
                print(f"✓ {test_name} test PASSED\n")
            else:
                print(f"✗ {test_name} test FAILED\n")
        except Exception as e:
            print(f"✗ {test_name} test FAILED with exception: {e}\n")
    
    print("Test Summary")
    print("============")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! Plugin is ready for use.")
        return 0
    else:
        print(f"✗ {total - passed} test(s) failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
