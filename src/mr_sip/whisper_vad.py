#!/usr/bin/env python3
"""
Whisper Streaming STT with Voice Activity Detection (VAD) for MindRoot SIP Plugin

Optimized version using faster-whisper for 4x faster transcription.
Detects pauses to identify complete utterances and reduce partials.
"""

import numpy as np
import threading
import queue
import time
import logging
from collections import deque
from typing import Callable, Optional

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

logger = logging.getLogger(__name__)

class WhisperStreamingVAD:
    """Faster-Whisper-based streaming speech-to-text with voice activity detection."""
    
    def __init__(self, model_size="base", sample_rate=16000, 
                 chunk_duration=0.15, silence_threshold=0.005, 
                 silence_duration=0.3, min_speech_duration=0.25,
                 utterance_callback: Optional[Callable] = None):
        """
        Initialize streaming Whisper transcriber with VAD
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            sample_rate: Audio sample rate in Hz
            chunk_duration: Duration of audio chunks for VAD analysis (default: 0.25s)
            silence_threshold: RMS threshold below which audio is considered silence
            silence_duration: Seconds of silence to trigger end of utterance (default: 0.3s)
            min_speech_duration: Minimum speech duration to process (filter out noise)
            utterance_callback: Callback function for complete utterances
                               Signature: callback(text, utterance_num, timestamp)
        """
        if WhisperModel is None:
            raise ImportError("faster-whisper not available. Install with: pip install faster-whisper")
            
        logger.info(f"Loading faster-whisper {model_size} model...")
        # Use CPU with 4 threads for good performance without GPU
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(sample_rate * chunk_duration)
        
        # VAD parameters - optimized for low latency
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.min_speech_duration = min_speech_duration
        self.silence_chunks_needed = int(silence_duration / chunk_duration)
        
        # State tracking
        self.audio_buffer = deque()
        self.speech_buffer = deque()
        self.is_speaking = False
        self.silence_counter = 0
        self.speech_start_time = None
        
        # Processing
        self.processing_queue = queue.Queue()
        self.is_running = False
        self.processing_thread = None
        self.last_transcription = ""
        self.utterance_count = 0
        
        # Callback for utterances
        self.utterance_callback = utterance_callback
        
        logger.info(f"WhisperStreamingVAD initialized (faster-whisper):")
        logger.info(f"  Model: {model_size}")
        logger.info(f"  Sample rate: {sample_rate} Hz")
        logger.info(f"  Chunk duration: {chunk_duration}s")
        logger.info(f"  Silence threshold: {silence_threshold}")
        logger.info(f"  Silence duration: {silence_duration}s")
        logger.info(f"  Min speech duration: {min_speech_duration}s")
        
    def start(self):
        """Start the processing thread"""
        if self.is_running:
            logger.warning("WhisperStreamingVAD already running")
            return
            
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.processing_thread.start()
        logger.info("Faster-whisper streaming transcription with VAD started")
        
    def stop(self):
        """Stop the processing thread"""
        if not self.is_running:
            return
            
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        logger.info("Faster-whisper streaming transcription stopped")
        
    def _calculate_rms(self, audio_chunk):
        """Calculate RMS (Root Mean Square) energy of audio chunk"""
        return np.sqrt(np.mean(audio_chunk**2))
    
    def _is_speech(self, audio_chunk):
        """Determine if audio chunk contains speech based on energy"""
        rms = self._calculate_rms(audio_chunk)
        return rms > self.silence_threshold
        
    def add_audio(self, audio_chunk):
        """
        Add audio chunk and perform VAD
        
        Args:
            audio_chunk: numpy array of audio samples (float32, mono)
        """
        if not self.is_running:
            return
            
        # Ensure proper format
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)
        if np.abs(audio_chunk).max() > 1.0:
            audio_chunk = audio_chunk / 32768.0
            
        audio_chunk = audio_chunk.flatten()
        self.audio_buffer.extend(audio_chunk)
        
        # Process in fixed-size chunks for VAD
        while len(self.audio_buffer) >= self.chunk_samples:
            chunk = np.array([self.audio_buffer.popleft() for _ in range(self.chunk_samples)])
            self._process_vad_chunk(chunk)
    
    def _process_vad_chunk(self, chunk):
        """Process a chunk for voice activity detection"""
        is_speech = self._is_speech(chunk)
        
        if is_speech:
            if not self.is_speaking:
                # Speech started
                self.is_speaking = True
                self.speech_start_time = time.time()
                self.silence_counter = 0
                logger.debug("[SPEECH START]")
            
            # Add to speech buffer
            self.speech_buffer.extend(chunk)
            self.silence_counter = 0
            
        else:
            # Silence detected
            if self.is_speaking:
                self.silence_counter += 1
                # Still add to buffer during silence (for context)
                self.speech_buffer.extend(chunk)
                
                # Check if silence duration threshold reached
                if self.silence_counter >= self.silence_chunks_needed:
                    # End of utterance
                    speech_duration = time.time() - self.speech_start_time
                    
                    if speech_duration >= self.min_speech_duration:
                        # Process the complete utterance
                        audio_array = np.array(list(self.speech_buffer))
                        self.processing_queue.put(('utterance', audio_array))
                        logger.debug(f"[SPEECH END] Duration: {speech_duration:.2f}s")
                    else:
                        logger.debug(f"[SPEECH IGNORED] Too short: {speech_duration:.2f}s")
                    
                    # Reset state
                    self.is_speaking = False
                    self.speech_buffer.clear()
                    self.silence_counter = 0
                    self.speech_start_time = None
    
    def _process_loop(self):
        """Background thread that processes complete utterances using faster-whisper"""
        logger.info("Faster-whisper processing loop started")
        
        while self.is_running:
            try:
                msg_type, audio_chunk = self.processing_queue.get(timeout=0.1)
                
                if msg_type == 'utterance':
                    # Transcribe the complete utterance using faster-whisper
                    try:
                        start_time = time.time()
                        
                        # faster-whisper returns an iterator of segments
                        segments, info = self.model.transcribe(
                            audio_chunk,
                            language=None,  # Auto-detect
                            beam_size=5,
                            vad_filter=False,  # We're doing our own VAD
                            without_timestamps=True  # Faster without timestamps
                        )
                        
                        # Collect all segments
                        text_parts = []
                        for segment in segments:
                            text_parts.append(segment.text)
                        
                        text = " ".join(text_parts).strip()
                        
                        transcribe_time = time.time() - start_time
                        
                        if text and text != self.last_transcription:
                            self.utterance_count += 1
                            timestamp = time.time()
                            
                            logger.info(f"Transcribed utterance #{self.utterance_count} in {transcribe_time:.2f}s: {text}")
                            
                            # Call the callback if provided
                            if self.utterance_callback:
                                try:
                                    self.utterance_callback(text, self.utterance_count, timestamp)
                                except Exception as e:
                                    logger.error(f"Error in utterance callback: {e}")
                            
                            self.last_transcription = text
                            
                    except Exception as e:
                        logger.error(f"Error transcribing audio: {e}")
                    
            except queue.Empty:
                continue
            except Exception as e:
                if self.is_running:
                    logger.error(f"Error in processing loop: {e}")
                continue
                
        logger.info("Faster-whisper processing loop stopped")

    def get_stats(self):
        """Get transcription statistics"""
        return {
            "utterance_count": self.utterance_count,
            "is_running": self.is_running,
            "is_speaking": self.is_speaking,
            "buffer_size": len(self.audio_buffer),
            "speech_buffer_size": len(self.speech_buffer)
        }
