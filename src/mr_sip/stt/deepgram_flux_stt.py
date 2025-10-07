#!/usr/bin/env python3
"""
Deepgram Flux STT Provider

Real-time conversational speech-to-text using Deepgram's Flux model.
Provides ultra-low latency turn detection with eager end-of-turn processing.
"""

import asyncio
import json
import logging
import time
import threading
import numpy as np
from typing import Optional, Callable
from deepgram import DeepgramClient
from deepgram.core.events import EventType
from .base_stt import BaseSTTProvider, STTResult

logger = logging.getLogger(__name__)

class DeepgramFluxSTT(BaseSTTProvider):
    """Deepgram Flux streaming STT provider with conversational turn detection."""
    
    def __init__(self, 
                 api_key: str,
                 sample_rate: int = 16000,
                 language: str = "en",
                 model: str = "flux-general-en",
                 eager_eot_threshold: float = 0.7,
                 eot_threshold: float = 0.8,
                 eot_timeout_ms: int = None):
        """
        Initialize Deepgram Flux STT provider.
        
        Args:
            api_key: Deepgram API key
            sample_rate: Audio sample rate in Hz (default: 16000)
            language: Language code (default: 'en')
            model: Deepgram model to use (default: 'flux-general-en')
            eager_eot_threshold: Threshold for EagerEndOfTurn events (0.3-0.9, default: 0.7)
            eot_threshold: Threshold for EndOfTurn events (default: 0.8)
            eot_timeout_ms: Turn timeout in milliseconds (optional)
        """
        super().__init__(sample_rate=sample_rate)
        self.api_key = api_key
        self.language = language
        self.model = model
        self.eager_eot_threshold = eager_eot_threshold
        self.eot_threshold = eot_threshold
        self.eot_timeout_ms = eot_timeout_ms
        
        # Deepgram client and connection
        self.client: Optional[DeepgramClient] = None
        self.connection = None
        self.connection_task = None
        
        # Processing state
        self.utterance_count = 0
        self.last_final_text = ""
        self.connection_start_time = None
        self.draft_response_active = False
        
        # Stats
        self.total_audio_sent = 0
        self.total_eager_eots = 0
        self.total_turn_resumed = 0
        self.total_finals = 0
        self.latencies = []
        
        # Turn resumed callback
        self._on_turn_resumed_callback: Optional[Callable[[], None]] = None
        
    async def start(self) -> None:
        """Connect to Deepgram Flux API."""
        if self.is_running:
            logger.warning("DeepgramFluxSTT already running")
            return
            
        try:
            logger.info(f"Connecting to Deepgram Flux API (model: {self.model})...")
            
            # Initialize Deepgram client
            self.client = DeepgramClient(api_key=self.api_key)
            
            # Build connection parameters
            connection_params = {
                "model": self.model,
                "encoding": "linear16",
                "sample_rate": self.sample_rate,
                "eager_eot_threshold": self.eager_eot_threshold,
                "eot_threshold": self.eot_threshold
            }
            
            # Add optional parameters
            if self.eot_timeout_ms is not None:
                connection_params["eot_timeout_ms"] = self.eot_timeout_ms
            
            # Connect to Flux using context manager pattern
            self.connection_task = asyncio.create_task(
                self._run_connection(**connection_params)
            )
            
            # Wait a moment for connection to establish
            await asyncio.sleep(0.5)
            
            self.is_running = True
            self.connection_start_time = time.time()
            
            logger.info("Connected to Deepgram Flux API")
            
        except Exception as e:
            logger.error(f"Failed to connect to Deepgram Flux: {e}")
            raise
            
    async def _run_connection(self, **connection_params):
        """Run the Deepgram connection in async context manager."""
        try:
            async with self.client.listen.v2.connect(**connection_params) as connection:
                self.connection = connection
                
                # Set up event handlers
                connection.on(EventType.OPEN, self._on_open)
                connection.on(EventType.MESSAGE, self._on_message)
                connection.on(EventType.ERROR, self._on_error)
                connection.on(EventType.CLOSE, self._on_close)
                
                # Start listening
                await connection.start_listening()
                
        except Exception as e:
            logger.error(f"Error in Deepgram connection: {e}")
            self.is_running = False
            
    async def stop(self) -> None:
        """Disconnect from Deepgram Flux API."""
        if not self.is_running:
            return
            
        self.is_running = False
        
        try:
            # Cancel the connection task
            if self.connection_task and not self.connection_task.done():
                self.connection_task.cancel()
                
            logger.info("Disconnected from Deepgram Flux API")
            
        except Exception as e:
            logger.error(f"Error stopping Deepgram Flux: {e}")
            
    async def add_audio(self, audio_chunk: np.ndarray) -> None:
        """Send audio chunk to Deepgram Flux."""
        if not self.is_running or not self.connection:
            logger.warning("Cannot send audio: not connected to Deepgram Flux")
            return
            
        try:
            # Ensure proper format
            if audio_chunk.dtype != np.float32:
                audio_chunk = audio_chunk.astype(np.float32)
                
            # Normalize to [-1, 1] if needed
            if np.abs(audio_chunk).max() > 1.0:
                audio_chunk = audio_chunk / 32768.0
                
            # Convert to 16-bit PCM
            audio_int16 = (audio_chunk * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            
            # Send to Deepgram Flux
            self.connection.send_media(audio_bytes)
            self.total_audio_sent += len(audio_bytes)
            
        except Exception as e:
            logger.error(f"Error sending audio to Deepgram Flux: {e}")
            
    def _on_open(self, *args) -> None:
        """Handle connection open event."""
        logger.debug("Deepgram Flux connection opened")
        
    def _on_close(self, *args) -> None:
        """Handle connection close event."""
        logger.info("Deepgram Flux connection closed")
        if self.is_running:
            # Unexpected close, try to reconnect
            logger.warning("Unexpected connection close, attempting reconnect...")
            asyncio.create_task(self._reconnect())
            
    def _on_error(self, error) -> None:
        """Handle connection error event."""
        logger.error(f"Deepgram Flux connection error: {error}")
        
    def _on_message(self, message) -> None:
        """Handle incoming message from Deepgram Flux."""
        try:
            # Check if this is a TurnInfo message
            if not hasattr(message, 'type') or message.type != 'TurnInfo':
                logger.debug(f"Received non-TurnInfo message: {getattr(message, 'type', 'unknown')}")
                return
                
            event = getattr(message, 'event', None)
            transcript = getattr(message, 'transcript', '').strip()
            
            if not transcript:
                return
                
            # Calculate latency
            latency = time.time() - self.connection_start_time if self.connection_start_time else 0
            self.latencies.append(latency)
            
            # Handle different event types
            if event == 'EagerEndOfTurn':
                self._handle_eager_eot(transcript, latency)
            elif event == 'TurnResumed':
                self._handle_turn_resumed(transcript, latency)
            elif event == 'EndOfTurn':
                self._handle_end_of_turn(transcript, latency)
            else:
                logger.debug(f"Unknown Flux event: {event}")
                
        except Exception as e:
            logger.error(f"Error handling Flux message: {e}")
            
    def _handle_eager_eot(self, transcript: str, latency: float) -> None:
        """Handle EagerEndOfTurn event - start preparing response."""
        self.total_eager_eots += 1
        self.draft_response_active = True
        
        logger.info(f"[EAGER EOT] {transcript} (latency: {latency*1000:.0f}ms)")
        
        # Create partial result for quick response preparation
        result = STTResult(
            text=transcript,
            is_final=False,  # This is a draft/partial  
            is_eager_eot=True,  # Flag for eager end of turn
            confidence=0.8,  # Medium confidence for eager EOT
            timestamp=time.time()
        )
        
        # Emit as partial for early processing
        self._emit_partial(result)
        
    def set_turn_resumed_callback(self, callback: Optional[Callable[[], None]]) -> None:
        """Set callback for TurnResumed events."""
        self._on_turn_resumed_callback = callback
        
    def _handle_turn_resumed(self, transcript: str, latency: float) -> None:
        """Handle TurnResumed event - cancel draft response."""
        self.total_turn_resumed += 1
        self.draft_response_active = False
        
        logger.info(f"[TURN RESUMED] User continued speaking (latency: {latency*1000:.0f}ms)")
        
        # Emit cancellation signal to SIP client
        if self._on_turn_resumed_callback:
            try:
                self._on_turn_resumed_callback()
            except Exception as e:
                logger.error(f"Error in turn resumed callback: {e}")
        
    def _handle_end_of_turn(self, transcript: str, latency: float) -> None:
        """Handle EndOfTurn event - finalize response."""
        self.utterance_count += 1
        self.total_finals += 1
        self.last_final_text = transcript
        self.draft_response_active = False
        
        logger.info(f"[FINAL #{self.utterance_count}] {transcript} (latency: {latency*1000:.0f}ms)")
        
        # Create final result
        result = STTResult(
            text=transcript,
            is_final=True,
            is_eager_eot=False,  # This is final, not eager
            confidence=0.95,  # High confidence for final EOT
            timestamp=time.time()
        )
        result.utterance_num = self.utterance_count
        
        # Emit final result
        self._emit_final(result)
        
    async def _reconnect(self) -> None:
        """Attempt to reconnect to Deepgram Flux."""
        if not self.is_running:
            return
            
        try:
            await asyncio.sleep(1.0)  # Brief delay before reconnect
            await self.stop()
            await self.start()
        except Exception as e:
            logger.error(f"Failed to reconnect to Deepgram Flux: {e}")
            
    def get_stats(self) -> dict:
        """Get statistics about the Deepgram Flux connection."""
        avg_latency = sum(self.latencies) / len(self.latencies) if self.latencies else 0
        
        return {
            "provider": "deepgram-flux",
            "model": self.model,
            "is_running": self.is_running,
            "utterance_count": self.utterance_count,
            "total_audio_sent_bytes": self.total_audio_sent,
            "total_eager_eots": self.total_eager_eots,
            "total_turn_resumed": self.total_turn_resumed,
            "total_finals": self.total_finals,
            "average_latency_ms": avg_latency * 1000,
            "connection_duration_seconds": time.time() - self.connection_start_time if self.connection_start_time else 0,
            "eager_eot_threshold": self.eager_eot_threshold,
            "eot_threshold": self.eot_threshold,
            "draft_response_active": self.draft_response_active
        }
