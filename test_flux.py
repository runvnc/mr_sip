#!/usr/bin/env python3
"""
Test script for Deepgram Flux STT integration
"""

import asyncio
import os
import sys
import logging
import numpy as np
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from mr_sip.stt import create_stt_provider

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_flux_stt():
    """Test the Deepgram Flux STT provider."""
    
    # Check for API key
    api_key = os.getenv('DEEPGRAM_API_KEY')
    if not api_key:
        logger.error("DEEPGRAM_API_KEY environment variable not set")
        return False
        
    try:
        # Create Flux STT provider
        logger.info("Creating Deepgram Flux STT provider...")
        stt = create_stt_provider(
            provider_name='deepgram_flux',
            api_key=api_key,
            eager_eot_threshold=0.6,  # Lower threshold for testing
            eot_threshold=0.8
        )
        
        # Set up event handlers
        def on_partial(result):
            logger.info(f"PARTIAL: {result.text} (confidence: {result.confidence:.2f})")
            
        def on_final(result):
            logger.info(f"FINAL: {result.text} (confidence: {result.confidence:.2f}, utterance: {result.utterance_num})")
            
        stt.on_partial_result = on_partial
        stt.on_final_result = on_final
        
        # Start the STT provider
        logger.info("Starting Flux STT provider...")
        await stt.start()
        
        # Generate some test audio (silence for now)
        logger.info("Sending test audio chunks...")
        sample_rate = 16000
        chunk_duration = 0.1  # 100ms chunks
        chunk_size = int(sample_rate * chunk_duration)
        
        # Send 5 seconds of silence (in real usage, this would be actual audio)
        for i in range(50):  # 50 chunks of 100ms = 5 seconds
            # Generate silence (zeros)
            audio_chunk = np.zeros(chunk_size, dtype=np.float32)
            await stt.add_audio(audio_chunk)
            await asyncio.sleep(0.1)  # Simulate real-time audio
            
        # Get stats
        stats = stt.get_stats()
        logger.info(f"STT Stats: {stats}")
        
        # Stop the provider
        logger.info("Stopping Flux STT provider...")
        await stt.stop()
        
        logger.info("Test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Testing Deepgram Flux STT Integration")
    print("=" * 40)
    
    success = asyncio.run(test_flux_stt())
    
    if success:
        print("\n✅ Test passed! Flux integration is working.")
    else:
        print("\n❌ Test failed. Check the logs above.")
        sys.exit(1)
