#!/usr/bin/env python3
"""
Baresip Integration for MindRoot

Adapted from baresip_transcriber.py to work with MindRoot's async architecture.
Provides SIP phone call functionality with real-time transcription and TTS output.
"""

import os
import sys
import time
import struct
import threading
import numpy as np
from scipy import signal
from queue import Empty
from baresipy import BareSIP
from datetime import datetime
from pathlib import Path
import asyncio
import logging
from typing import Callable, Optional, Any

# Import the VAD transcriber from whispertest
try:
    sys.path.append('/files/whispertest')
    from whisper_streaming_vad import WhisperStreamingVAD
except ImportError:
    print("Error: whisper_streaming_vad.py not found!")
    print("Make sure /files/whispertest/whisper_streaming_vad.py exists.")
    WhisperStreamingVAD = None

logger = logging.getLogger(__name__)

class MindRootSIPBot(BareSIP):
    """
    SIP phone bot integrated with MindRoot's AI agent system.
    
    Features:
    - Captures incoming audio using baresip's sndfile module
    - Processes audio in real-time with VAD
    - Sends transcribed utterances to MindRoot agent
    - Receives TTS audio from MindRoot and plays it on the call
    """
    
    def __init__(self, user, password, gateway, audio_dir=".", 
                 on_utterance_callback=None, model_size="small", context=None):
        """
        Args:
            user: SIP username
            password: SIP password
            gateway: SIP gateway
            audio_dir: Directory where baresip saves recordings
            on_utterance_callback: Async function called with each complete utterance
                                  Signature: async callback(text, utterance_num, timestamp, context)
            model_size: Whisper model size (tiny, base, small, medium, large)
            context: MindRoot ChatContext
        """
        # Set up audio directory
        self.audio_dir = audio_dir or os.path.expanduser("~/.baresip")
        
        # Initialize baresipy
        super().__init__(user, password, gateway, block=False)
        
        # MindRoot integration
        self.context = context
        self.on_utterance_callback = on_utterance_callback
        
        # Call tracking
        self.call_start_time = None
        self.current_dec_file = None  # Incoming audio file
        self.current_enc_file = None  # Outgoing audio file
        
        # Transcription
        self.transcriber = None
        self.model_size = model_size
        self.utterances = []
        
        # Processing thread
        self.audio_processor_thread = None
        self.processing = False
        
        # TTS audio output
        self.tts_audio_queue = asyncio.Queue()
        self.tts_sender_task = None
        
    async def _on_utterance(self, text, utterance_num, timestamp):
        """Internal callback for utterances - sends to MindRoot agent"""
        utterance_data = {
            'number': utterance_num,
            'text': text,
            'timestamp': timestamp,
            'time_str': time.strftime("%H:%M:%S", time.localtime(timestamp))
        }
        self.utterances.append(utterance_data)
        
        logger.info(f"[{utterance_data['time_str']}] Utterance #{utterance_num}: {text}")
        
        # Call MindRoot callback if provided
        if self.on_utterance_callback:
            try:
                if asyncio.iscoroutinefunction(self.on_utterance_callback):
                    await self.on_utterance_callback(text, utterance_num, timestamp, self.context)
                else:
                    # Run sync callback in thread pool
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        None, 
                        self.on_utterance_callback, 
                        text, utterance_num, timestamp, self.context
                    )
            except Exception as e:
                logger.error(f"Error in utterance callback: {e}")
        
    def handle_call_established(self):
        """When call connects, start transcription"""
        logger.info("=== CALL ESTABLISHED ===")
        self.call_start_time = datetime.now()
        
        # Initialize transcriber if available
        if WhisperStreamingVAD:
            logger.info(f"Initializing Whisper ({self.model_size} model)...")
            self.transcriber = WhisperStreamingVAD(
                model_size=self.model_size,
                sample_rate=16000,  # Whisper expects 16kHz
                chunk_duration=0.5,
                silence_threshold=0.01,    # Adjust for phone line noise
                silence_duration=1.0,      # 1 second pause = end of utterance
                min_speech_duration=0.5    # Ignore very short sounds
            )
            
            # Setup transcriber callback
            self._setup_transcriber_callback()
            
            # Start transcriber
            self.transcriber.start()
            logger.info("Transcriber started.")
        else:
            logger.warning("WhisperStreamingVAD not available, transcription disabled")
        
        # Wait for audio file to be created
        time.sleep(0.5)
        self._find_current_audio_files()
        
        if self.current_dec_file:
            logger.info(f"Found incoming audio file: {self.current_dec_file}")
            logger.info("Starting real-time transcription...")
            
            # Start processing incoming audio
            self.processing = True
            self.audio_processor_thread = threading.Thread(
                target=self._process_incoming_audio_realtime,
                daemon=True
            )
            self.audio_processor_thread.start()
        else:
            logger.warning("Could not find audio recording files.")
            logger.warning("Make sure sndfile module is enabled in ~/.baresip/config")
            
        # Start TTS audio sender
        if asyncio.get_event_loop().is_running():
            self.tts_sender_task = asyncio.create_task(self._tts_audio_sender())
            
    def _setup_transcriber_callback(self):
        """Setup callback to capture complete utterances"""
        if not self.transcriber:
            return
            
        original_process = self.transcriber._process_loop
        
        def wrapped_process():
            while self.transcriber.is_running:
                try:
                    msg_type, audio_chunk = self.transcriber.processing_queue.get(timeout=0.1)
                    
                    if msg_type == 'utterance':
                        result = self.transcriber.model.transcribe(
                            audio_chunk,
                            language=None,
                            fp16=False,
                            verbose=False
                        )
                        
                        text = result["text"].strip()
                        if text and text != self.transcriber.last_transcription:
                            self.transcriber.utterance_count += 1
                            timestamp = time.time()
                            
                            # Call our async callback
                            if asyncio.get_event_loop().is_running():
                                asyncio.create_task(
                                    self._on_utterance(
                                        text,
                                        self.transcriber.utterance_count,
                                        timestamp
                                    )
                                )
                            
                            self.transcriber.last_transcription = text
                        
                except Empty:
                    continue
                except Exception as e:
                    if self.transcriber.is_running:
                        logger.error(f"Error processing audio: {e}")
                    continue
        
        self.transcriber._process_loop = wrapped_process
            
    def _find_current_audio_files(self):
        """Find the audio files that baresip just created for this call"""
        now = time.time()
        
        for filename in os.listdir(self.audio_dir):
            if filename.startswith("dump-") and filename.endswith(".wav"):
                filepath = os.path.join(self.audio_dir, filename)
                file_time = os.path.getmtime(filepath)
                
                # If file was created in the last 5 seconds
                if now - file_time < 5:
                    if "-dec.wav" in filename:
                        self.current_dec_file = filepath
                    elif "-enc.wav" in filename:
                        self.current_enc_file = filepath
    
    def _parse_wav_header(self, filepath):
        """Parse WAV header to get audio format info"""
        try:
            with open(filepath, 'rb') as f:
                # Read RIFF header
                riff = f.read(4)
                if riff != b'RIFF':
                    return None
                    
                file_size = struct.unpack('<I', f.read(4))[0]
                wave_tag = f.read(4)
                if wave_tag != b'WAVE':
                    return None
                
                # Find fmt chunk
                while True:
                    chunk_id = f.read(4)
                    if not chunk_id:
                        return None
                    chunk_size = struct.unpack('<I', f.read(4))[0]
                    
                    if chunk_id == b'fmt ':
                        fmt_data = f.read(chunk_size)
                        channels = struct.unpack('<H', fmt_data[2:4])[0]
                        sample_rate = struct.unpack('<I', fmt_data[4:8])[0]
                        bits_per_sample = struct.unpack('<H', fmt_data[14:16])[0]
                        sample_width = bits_per_sample // 8
                        break
                    else:
                        f.seek(chunk_size, 1)
                
                # Find data chunk
                while True:
                    chunk_id = f.read(4)
                    if not chunk_id:
                        return None
                    chunk_size = struct.unpack('<I', f.read(4))[0]
                    
                    if chunk_id == b'data':
                        data_offset = f.tell()
                        return (channels, sample_width, sample_rate, data_offset)
                    else:
                        f.seek(chunk_size, 1)
                        
        except Exception as e:
            logger.error(f"Error parsing WAV header: {e}")
            return None
                        
    def _process_incoming_audio_realtime(self):
        """Process incoming audio in real-time and feed to transcriber"""
        if not self.current_dec_file or not self.transcriber:
            return
            
        logger.info(f"Processing audio from: {self.current_dec_file}")
        
        # Wait for file to be ready
        max_wait_attempts = 50
        wait_count = 0
        while (not os.path.exists(self.current_dec_file) or 
               os.path.getsize(self.current_dec_file) < 100) and \
              wait_count < max_wait_attempts:
            time.sleep(0.1)
            wait_count += 1
            if not self.call_established:
                return
                
        if wait_count >= max_wait_attempts:
            logger.error("Timeout waiting for audio file to be ready")
            return
                
        try:
            # Wait for WAV header to be written
            time.sleep(1.0)
            
            # Parse header
            wav_info = self._parse_wav_header(self.current_dec_file)
            if not wav_info:
                logger.error("Failed to parse WAV header")
                return
                
            channels, sample_width, framerate, data_offset = wav_info
            logger.info(f"Audio format: {channels} channels, {sample_width} bytes/sample, {framerate} Hz")
            
            # Check if we need to resample
            needs_resampling = framerate != 16000
            if needs_resampling:
                logger.info(f"Audio will be resampled from {framerate}Hz to 16000Hz for Whisper")
                resample_ratio = 16000 / framerate
            else:
                logger.info("Audio is already at 16kHz, no resampling needed")
                resample_ratio = 1.0
            
            # Open file and read audio data
            with open(self.current_dec_file, 'rb') as f:
                f.seek(data_offset)
                
                chunk_size = 8000 * sample_width * channels  # ~0.5 seconds at 16kHz
                empty_read_count = 0
                max_empty_reads = 100
                
                logger.info("Starting real-time transcription...")
                
                while self.processing and self.call_established:
                    try:
                        audio_data = f.read(chunk_size)
                        
                        if audio_data:
                            empty_read_count = 0
                            
                            # Convert to numpy array
                            audio_array = np.frombuffer(audio_data, dtype=np.int16)
                            
                            # Convert to float32 in range [-1, 1]
                            audio_float = audio_array.astype(np.float32) / 32768.0
                            
                            # Resample if needed (8kHz -> 16kHz for phone audio)
                            if needs_resampling:
                                num_samples = int(len(audio_float) * resample_ratio)
                                audio_float = signal.resample(audio_float, num_samples)
                            
                            # Feed to transcriber
                            self.transcriber.add_audio(audio_float)
                        else:
                            # No more data yet, wait
                            empty_read_count += 1
                            if empty_read_count > max_empty_reads:
                                empty_read_count = 0
                            time.sleep(0.01)
                            
                    except Exception as e:
                        logger.error(f"Error reading audio data: {e}")
                        time.sleep(0.1)
                        
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            
    async def _tts_audio_sender(self):
        """Background task to send TTS audio to the call"""
        logger.info("TTS audio sender started")
        try:
            while self.call_established:
                try:
                    audio_chunk = await asyncio.wait_for(self.tts_audio_queue.get(), timeout=1.0)
                    if audio_chunk is None:  # Sentinel to stop
                        break
                    await self._send_tts_audio(audio_chunk)
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Error in TTS audio sender: {e}")
                    break
        except asyncio.CancelledError:
            logger.info("TTS audio sender cancelled")
        finally:
            logger.info("TTS audio sender stopped")
            
    async def _send_tts_audio(self, audio_chunk: bytes):
        """Send TTS audio chunk to the SIP call"""
        # TODO: Implement actual audio injection into baresip call
        # This is a complex task that may require:
        # 1. Converting audio format to match call requirements
        # 2. Writing to baresip's audio input stream
        # 3. Or using baresip's audio injection APIs if available
        
        logger.debug(f"Received TTS audio chunk: {len(audio_chunk)} bytes")
        # For now, just log that we received the audio
        # In a full implementation, this would inject the audio into the call
        
    async def send_tts_audio(self, audio_chunk: bytes):
        """Queue TTS audio chunk for sending to the call"""
        if self.call_established:
            try:
                await self.tts_audio_queue.put(audio_chunk)
            except Exception as e:
                logger.error(f"Failed to queue TTS audio: {e}")
                
    def handle_call_ended(self, reason):
        """When call ends, stop processing and cleanup"""
        logger.info("=== CALL ENDED ===")
        self.processing = False
        
        # Stop transcriber
        if self.transcriber:
            time.sleep(1)  # Let final utterance process
            self.transcriber.stop()
        
        # Stop TTS sender
        if self.tts_sender_task:
            self.tts_sender_task.cancel()
        
        # Show summary
        if self.current_dec_file and os.path.exists(self.current_dec_file):
            size = os.path.getsize(self.current_dec_file)
            logger.info(f"Incoming audio saved to: {self.current_dec_file}")
            logger.info(f"File size: {size:,} bytes")
            
        # Show transcript summary
        if self.utterances:
            logger.info(f"Call transcript: {len(self.utterances)} utterances captured")
            for utterance in self.utterances[-3:]:  # Show last 3 utterances
                logger.info(f"[{utterance['time_str']}] {utterance['text']}")
        else:
            logger.info("No utterances detected.")
            
        super().handle_call_ended(reason)
    
    def get_transcript(self):
        """Get full transcript as a single string"""
        return "\n".join([u['text'] for u in self.utterances])
    
    def get_utterances(self):
        """Get all captured utterances"""
        return self.utterances

def setup_sndfile_module():
    """Helper function to set up the sndfile module in baresip config"""
    config_path = os.path.expanduser("~/.baresip/config")
    
    if not os.path.exists(config_path):
        logger.warning("Baresip config not found. Run baresip once to create it.")
        return False
        
    with open(config_path, 'r') as f:
        config = f.read()
        
    if "module\t\t\tsndfile.so" in config:
        logger.info("sndfile module already enabled")
        return True
    elif "#module\t\t\tsndfile.so" in config:
        config = config.replace("#module\t\t\tsndfile.so", "module\t\t\tsndfile.so")
        with open(config_path, 'w') as f:
            f.write(config)
        logger.info("Enabled sndfile module in config")
        return True
    else:
        if "module_path" in config:
            config = config.replace(
                "module_path\t\t/usr/local/lib/baresip/modules",
                "module_path\t\t/usr/local/lib/baresip/modules\nmodule\t\t\tsndfile.so"
            )
            with open(config_path, 'w') as f:
                f.write(config)
            logger.info("Added sndfile module to config")
            return True
        else:
            logger.warning("Could not automatically enable sndfile module.")
            logger.warning(f"Please add 'module\t\t\tsndfile.so' to {config_path}")
            return False
