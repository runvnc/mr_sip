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
import sys
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
        
        # Connection health monitoring
        self.connection_healthy = False
        self.last_message_time = None
        self.connection_timeout = 30.0  # seconds
        
        # Stats
        self.total_audio_sent = 0
        self.total_eager_eots = 0
        self.total_turn_resumed = 0
        self.total_finals = 0
        self.latencies = []
        
        # Audio buffering for reconnection
        self.audio_buffer = []
        
        # Store reference to main event loop for cross-thread task scheduling
        self.main_loop = None
        try:
            self.main_loop = asyncio.get_running_loop()
        except RuntimeError:
            self.main_loop = None
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 3
        self.total_reconnections = 0
        
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
            
            # Set running flag before starting connection
            self.is_running = True
            self.connection_start_time = time.time()
            
            # Connect to Flux using synchronous context manager pattern
            self.connection_task = asyncio.create_task(
                self._run_connection(**connection_params)
            )
            
            # Start connection health monitor
            self.health_monitor_task = asyncio.create_task(
                self._monitor_connection_health()
            )
            # Wait a moment for connection to establish
            await asyncio.sleep(0.5)
            
            logger.info("Connected to Deepgram Flux API")
            
        except Exception as e:
            logger.error(f"Failed to connect to Deepgram Flux: {e}")
            raise
            
    async def _monitor_connection_health(self):
        """Monitor connection health and exit if connection fails."""
        await asyncio.sleep(5.0)  # Give connection time to establish
        
        while self.is_running:
            try:
                # Check if connection was ever established
                if not self.connection_healthy and self.connection_start_time:
                    elapsed = time.time() - self.connection_start_time
                    if elapsed > 10.0:  # 10 seconds to establish
                        logger.error("Deepgram Flux connection failed to establish after 10 seconds")
                        self._exit_process("Connection establishment timeout")
                        return
                
                # Check for message timeout (no messages received)
                if self.connection_healthy and self.last_message_time:
                    elapsed = time.time() - self.last_message_time
                    if elapsed > self.connection_timeout:
                        logger.error(f"No messages from Deepgram Flux for {elapsed:.1f} seconds")
                        self._exit_process("Message timeout")
                        return
                        
                await asyncio.sleep(5.0)  # Check every 5 seconds
            except Exception as e:
                logger.error(f"Error in connection health monitor: {e}")
                await asyncio.sleep(1.0)
                
    async def _run_connection(self, **connection_params):
        """Run the Deepgram connection in synchronous context manager."""
        logger.info(f"Starting Deepgram connection with params: {connection_params}")
        try:
            with self.client.listen.v2.connect(**connection_params) as connection:
                self.connection = connection
                
                # Set up event handlers
                connection.on(EventType.OPEN, self._on_open)
                connection.on(EventType.MESSAGE, self._on_message)
                connection.on(EventType.ERROR, self._on_error)
                connection.on(EventType.CLOSE, self._on_close)
                
                # Start listening in thread (as shown in example)
                import threading
                listen_thread = threading.Thread(target=connection.start_listening, daemon=True)
                listen_thread.start()
                
                # Keep the connection alive
                logger.info("Entering connection keep-alive loop")
                while self.is_running:
                    await asyncio.sleep(0.1)
                logger.info("Exiting connection keep-alive loop")
                
        except Exception as e:
            logger.error(f"Error in Deepgram connection: {e}")
            self.is_running = False
            self._exit_process(f"Connection error: {e}")
            
    def _exit_process(self, reason: str):
        """Exit the process due to connection failure."""
        logger.error(f"EXITING PROCESS: Deepgram Flux connection failed - {reason}")
        sys.exit(1)
            
    async def stop(self) -> None:
        """Disconnect from Deepgram Flux API."""
        if not self.is_running:
            return
            
        self.is_running = False
        
        try:
            # Cancel the connection task
            if self.connection_task and not self.connection_task.done():
                self.connection_task.cancel()
                
            # Cancel health monitor
            if hasattr(self, 'health_monitor_task') and not self.health_monitor_task.done():
                self.health_monitor_task.cancel()
                
            logger.info("Disconnected from Deepgram Flux API")
            
        except Exception as e:
            logger.error(f"Error stopping Deepgram Flux: {e}")
            
    async def add_audio(self, audio_chunk: np.ndarray) -> None:
        """Send audio chunk to Deepgram Flux."""
        # Always buffer audio (keep last 10 chunks to prevent memory growth)
        self.audio_buffer.append(audio_chunk.copy())
        if len(self.audio_buffer) > 10:
            self.audio_buffer.pop(0)
            
        if not self.is_running or not self.connection:
            logger.debug("Cannot send audio: not connected to Deepgram Flux")
            # Always attempt reconnection
            logger.warning("🔄 TRIGGERING BUFFERED RECONNECTION - Connection lost!")
            self._schedule_coroutine_threadsafe(self._reconnect_with_buffer())
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
            logger.debug(f"Sent {len(audio_bytes)} bytes to Deepgram Flux (total: {self.total_audio_sent})")
            
        except Exception as e:
            # Filter out normal WebSocket status messages that aren't actually errors
            error_msg = str(e)
            if "sent" in error_msg and "received" in error_msg and "OK" in error_msg:
                logger.debug(f"WebSocket status (not an error): {e}")
            else:
                logger.error(f"Error sending audio to Deepgram Flux: {e}")
                # Always attempt reconnection on send error
                logger.warning("🔄 TRIGGERING BUFFERED RECONNECTION after send error!")
                self._schedule_coroutine_threadsafe(self._reconnect_with_buffer())
                
    def _on_open(self, *args) -> None:
        """Handle connection open event."""
        logger.info("Deepgram Flux connection opened - ready to receive audio")
        self.connection_healthy = True
        logger.debug("Deepgram Flux connection opened")
        
    def _on_close(self, *args) -> None:
        """Handle connection close event."""
        logger.info("Deepgram Flux connection closed")
        if self.is_running:
            # Unexpected close, try to reconnect with buffer
            self.connection_healthy = False
            logger.warning("🔄 TRIGGERING BUFFERED RECONNECTION due to unexpected close")
            self._schedule_coroutine_threadsafe(self._reconnect_with_buffer())
            
    def _on_error(self, error) -> None:
        """Handle connection error event."""
        error_str = str(error)
        logger.error(f"Deepgram Flux connection error: {error}")
        
        # Handle 1011 websocket errors and other connection issues
        if "1011" in error_str or "policy violation" in error_str.lower() or "connection" in error_str.lower():
            logger.warning(f"🔄 TRIGGERING BUFFERED RECONNECTION due to error: {error}")
            self._schedule_coroutine_threadsafe(self._reconnect_with_buffer())
        
    def _on_message(self, message) -> None:
        """Handle incoming message from Deepgram Flux."""
        logger.info(f"Received message from Deepgram Flux: {getattr(message, 'type', 'unknown')}")
        self.last_message_time = time.time()
        
        try:
            # Check if this is a TurnInfo message
            if not hasattr(message, 'type') or message.type != 'TurnInfo':
                logger.debug(f"Received non-TurnInfo message: {getattr(message, 'type', 'unknown')}")
                return
                
            event = getattr(message, 'event', None)
            transcript = getattr(message, 'transcript', '').strip()
            
            if not transcript:
                logger.debug(f"Received TurnInfo message with empty transcript, event: {event}")
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
        
    def _should_attempt_reconnect(self) -> bool:
        """Check if we should attempt reconnection."""
        # Don't reconnect if we don't have a proper SIP context yet
        # This prevents reconnection during call setup phase
        if not hasattr(self, 'sip_call_established') or not getattr(self, 'sip_call_established', False):
            logger.warning("🔍 DEBUG: Skipping reconnection - SIP call not established yet")
            return False
            
        # Always allow reconnection attempts, just check attempt limits
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.warning(f"Max reconnection attempts reached: {self.reconnect_attempts}/{self.max_reconnect_attempts}")
            return False
            
        return True
        
    def set_sip_call_established(self, established: bool):
        """Set whether the SIP call is established (prevents premature reconnection)."""
        self.sip_call_established = established
        logger.error(f"🔍 DEBUG: SIP call established status set to: {established}")
        
    async def _reconnect_with_buffer(self):
        """Reconnect and immediately send buffered audio."""
        if not self._should_attempt_reconnect():
            return
            
        try:
            self.reconnect_attempts += 1
            self.total_reconnections += 1
            
            logger.error("=" * 80)
            logger.error(f"BUFFERED RECONNECTION #{self.total_reconnections} (attempt {self.reconnect_attempts}/{self.max_reconnect_attempts})")
            
            # Smart backoff based on audio availability
            if len(self.audio_buffer) == 0:
                # No audio = call setup/silence, use slower backoff
                backoff_times = [1.0, 2.0, 5.0]
                backoff_time = backoff_times[min(self.reconnect_attempts - 1, len(backoff_times) - 1)]
                logger.error(f"NO AUDIO BUFFERED - Using slow backoff: {backoff_time}s")
            else:
                # Have audio = active conversation, use fast backoff  
                backoff_times = [0.1, 0.2, 0.5]
                backoff_time = backoff_times[min(self.reconnect_attempts - 1, len(backoff_times) - 1)]
                logger.error(f"AUDIO BUFFERED ({len(self.audio_buffer)} chunks) - Using fast backoff: {backoff_time}s")
                
            logger.error("=" * 80)
            
            if backoff_time > 0:
                logger.warning(f"Waiting {backoff_time}s before reconnection...")
                await asyncio.sleep(backoff_time)
            
            logger.warning(f"Starting reconnection to Deepgram Flux...")
            
            # Restart connection
            await self.stop()
            await self.start()
            
            # Wait a moment for connection to stabilize
            await asyncio.sleep(0.2)
            
            # Send buffered audio if available
            if len(self.audio_buffer) > 0:
                logger.warning(f"Sending {len(self.audio_buffer)} buffered audio chunks immediately...")
                for i, chunk in enumerate(self.audio_buffer):
                    if self.connection and self.connection_healthy:
                        await self._send_audio_chunk_direct(chunk)
                        await asyncio.sleep(0.01)  # Small delay between chunks
                    else:
                        logger.error(f"Connection lost during buffer send at chunk {i}")
                        break
            else:
                logger.warning("No buffered audio - connection ready for new audio")
                    
            # Reset on successful reconnection
            logger.warning(f"RECONNECTION COMPLETE - resuming normal operation")
            self.reconnect_attempts = 0
            
        except Exception as e:
            logger.error(f"RECONNECTION FAILED: {e}")
            
    async def _send_audio_chunk_direct(self, audio_chunk: np.ndarray) -> None:
        """Send audio chunk directly to Deepgram (for buffered sending)."""
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
        except Exception as e:
            logger.error(f"Error in direct audio send: {e}")
            
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
            "draft_response_active": self.draft_response_active,
            "total_reconnections": self.total_reconnections,
            "current_reconnect_attempts": self.reconnect_attempts,
            "audio_buffer_size": len(self.audio_buffer)
        }
        
    def _schedule_coroutine_threadsafe(self, coro):
        """Schedule a coroutine to run in the main event loop from any thread."""
        if self.main_loop and not self.main_loop.is_closed():
            try:
                future = asyncio.run_coroutine_threadsafe(coro, self.main_loop)
                return future
            except Exception as e:
                logger.error(f"Failed to schedule coroutine from thread: {e}")
        else:
            logger.warning("No main event loop available for cross-thread scheduling")
        return None
