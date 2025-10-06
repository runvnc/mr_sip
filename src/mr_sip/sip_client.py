#!/usr/bin/env python3
"""
SIP Client for MindRoot
Core SIP functionality extracted from baresip_integration.py
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
from .audio_handler import AudioHandler
from .whisper_vad import WhisperStreamingVAD

logger = logging.getLogger(__name__)

class MindRootSIPBot(BareSIP):
    """SIP phone bot integrated with MindRoot's AI agent system."""
    
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
        self.current_dec_file = None
        self.current_enc_file = None
        
        # Transcription
        self.transcriber = None
        self.model_size = model_size
        self.utterances = []
        
        # Audio processing
        self.audio_processor_thread = None
        self.processing = False
        
        # NEW: Audio handler with JACK integration
        self.audio_handler = AudioHandler()
        
        # TTS audio output
        self.tts_audio_queue = None  # Will be created when needed
        self.tts_sender_task = None
        
        # Store reference to main event loop for cross-thread task scheduling
        try:
            self.main_loop = asyncio.get_running_loop()
        except RuntimeError:
            self.main_loop = None
        
    def _schedule_coroutine(self, coro):
        """Schedule a coroutine to run in the main event loop from any thread."""
        if self.main_loop and not self.main_loop.is_closed():
            try:
                # Use call_soon_threadsafe to schedule from another thread
                future = asyncio.run_coroutine_threadsafe(coro, self.main_loop)
                return future
            except Exception as e:
                logger.error(f"Failed to schedule coroutine: {e}")
        return None
        
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
        """When call connects, setup JACK and start transcription"""
        logger.info("=== CALL ESTABLISHED ===")
        self.call_start_time = datetime.now()
        
        # Setup JACK audio output
        if not self.audio_handler.jack_enabled:
            self.audio_handler.setup_jack_audio()
            self.audio_handler.configure_baresip_jack(self)
            
        # Connect JACK ports after call is established
        time.sleep(1.0)  # Give baresip time to create ports
        self.audio_handler.connect_jack_to_baresip()
        
        # Setup transcription
        self._setup_transcription()
        
        # Wait for audio file to be created
        time.sleep(11.5)
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
            
        # Start TTS audio sender using thread-safe scheduling
        if self.main_loop and not self.main_loop.is_closed():
            try:
                # Create the queue in the main loop if it doesn't exist
                if self.tts_audio_queue is None:
                    future = asyncio.run_coroutine_threadsafe(
                        self._create_tts_queue(), self.main_loop
                    )
                    future.result(timeout=5.0)  # Wait up to 5 seconds
                
                # Schedule the TTS sender task
                future = asyncio.run_coroutine_threadsafe(
                    self._start_tts_sender(), self.main_loop
                )
                logger.info("TTS audio sender scheduled")
            except Exception as e:
                logger.error(f"Failed to start TTS audio sender: {e}")
        else:
            logger.warning("No main event loop available for TTS audio sender")
            
    async def _create_tts_queue(self):
        """Create the TTS audio queue in the main event loop."""
        if self.tts_audio_queue is None:
            self.tts_audio_queue = asyncio.Queue()
            logger.debug("TTS audio queue created")
            
    async def _start_tts_sender(self):
        """Start the TTS audio sender task."""
        if self.tts_sender_task is None or self.tts_sender_task.done():
            self.tts_sender_task = asyncio.create_task(self._tts_audio_sender())
            logger.debug("TTS audio sender task started")
            
    def _setup_transcription(self):
        """Setup Whisper transcription."""
        try:
            logger.info(f"Initializing integrated Whisper ({self.model_size} model)...")
            
            # Create utterance callback that schedules async processing
            def utterance_callback(text, utterance_num, timestamp):
                """Sync callback that schedules async processing"""
                self._schedule_coroutine(
                    self._on_utterance(text, utterance_num, timestamp)
                )
            
            self.transcriber = WhisperStreamingVAD(
                model_size=self.model_size,
                sample_rate=16000,  # Whisper expects 16kHz
                chunk_duration=0.5,
                silence_threshold=0.01,    # Adjust for phone line noise
                silence_duration=1.0,      # 1 second pause = end of utterance
                min_speech_duration=0.5,   # Ignore very short sounds
                utterance_callback=utterance_callback
            )
            
            # Start transcriber
            self.transcriber.start()
            logger.info("Integrated Whisper transcriber started.")
            
        except ImportError as e:
            logger.error(f"Whisper not available: {e}")
            logger.error("Install with: pip install openai-whisper")
            self.transcriber = None
        except Exception as e:
            logger.error(f"Failed to initialize Whisper: {e}")
            self.transcriber = None
            
    def _find_current_audio_files(self):
        """Find the audio files that baresip just created for this call"""
        # Wait 5 seconds for files to be created and written
        logger.info("Waiting 5 seconds for audio files to be created...")
        time.sleep(5)
        
        # the audio dir is actually always just the working dir
        self.audio_dir = os.getcwd()
        logger.info(f"Looking for audio files in: {self.audio_dir}")
        
        # Find all dump files and get the most recently modified ones
        dec_files = []
        enc_files = []
        
        for filename in os.listdir(self.audio_dir):
            if filename.startswith("dump-") and filename.endswith(".wav"):
                logger.debug(f"Found audio file: {filename}")
                filepath = os.path.join(self.audio_dir, filename)
                file_time = os.path.getmtime(filepath)
                
                if "-dec.wav" in filename:
                    dec_files.append((filepath, file_time))
                elif "-enc.wav" in filename:
                    enc_files.append((filepath, file_time))
        
        # Pick the most recently modified files
        if dec_files:
            dec_files.sort(key=lambda x: x[1], reverse=True)  # Sort by modification time, newest first
            self.current_dec_file = dec_files[0][0]
            logger.info(f"Selected most recent decoded audio file: {self.current_dec_file}")
        
        if enc_files:
            enc_files.sort(key=lambda x: x[1], reverse=True)  # Sort by modification time, newest first
            self.current_enc_file = enc_files[0][0]
            logger.info(f"Selected most recent encoded audio file: {self.current_enc_file}")    
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
                    if self.tts_audio_queue is None:
                        await asyncio.sleep(0.1)
                        continue
                        
                    audio_chunk = await asyncio.wait_for(self.tts_audio_queue.get(), timeout=1.0)
                    if audio_chunk is None:  # Sentinel to stop
                        break
                    await self.send_tts_audio(audio_chunk)
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Error in TTS audio sender: {e}")
                    break
        except asyncio.CancelledError:
            logger.info("TTS audio sender cancelled")
        finally:
            logger.info("TTS audio sender stopped")
            
    async def send_tts_audio(self, audio_chunk: bytes):
        """Send TTS audio chunk to the call via JACK."""
        await self.audio_handler.send_tts_audio(audio_chunk)
                
    async def queue_tts_audio(self, audio_chunk: bytes):
        """Queue TTS audio chunk for sending to the call"""
        if self.call_established and self.tts_audio_queue is not None:
            try:
                await self.tts_audio_queue.put(audio_chunk)
            except Exception as e:
                logger.error(f"Failed to queue TTS audio: {e}")
                
    def handle_call_ended(self, reason):
        """When call ends, cleanup JACK and transcription"""
        logger.info("=== CALL ENDED ===")
        self.processing = False
        
        # Stop transcriber
        if self.transcriber:
            time.sleep(1)  # Let final utterance process
            self.transcriber.stop()
        
        # Stop TTS sender
        if self.tts_sender_task:
            self.tts_sender_task.cancel()
        
        # Cleanup audio handler
        self.audio_handler.cleanup(self)
        
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
