#!/usr/bin/env python3
"""
Deepgram STT Provider

Real-time streaming speech-to-text using Deepgram's WebSocket API.
Provides low-latency partial and final transcriptions.
"""

import asyncio
import json
import logging
import time
import numpy as np
from typing import Optional
import websockets
from .base_stt import BaseSTTProvider, STTResult

logger = logging.getLogger(__name__)

class DeepgramSTT(BaseSTTProvider):
    """Deepgram streaming STT provider."""
    
    def __init__(self, 
                 api_key: str,
                 sample_rate: int = 16000,
                 language: str = "en-US",
                 model: str = "nova-2",
                 interim_results: bool = True,
                 punctuate: bool = True,
                 smart_format: bool = True):
        """
        Initialize Deepgram STT provider.
        
        Args:
            api_key: Deepgram API key
            sample_rate: Audio sample rate in Hz (default: 16000)
            language: Language code (default: 'en-US')
            model: Deepgram model to use (default: 'nova-2' - latest and best)
            interim_results: Enable partial/interim results (default: True)
            punctuate: Add punctuation (default: True)
            smart_format: Apply smart formatting (default: True)
        """
        super().__init__(sample_rate=sample_rate)
        self.api_key = api_key
        self.language = language
        self.model = model
        self.interim_results = interim_results
        self.punctuate = punctuate
        self.smart_format = smart_format
        
        # WebSocket connection
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.ws_url = self._build_ws_url()
        
        # Processing state
        self.utterance_count = 0
        self.last_final_text = ""
        self.connection_start_time = None
        
        # Stats
        self.total_audio_sent = 0
        self.total_partials = 0
        self.total_finals = 0
        self.latencies = []
        
        # Tasks
        self.receive_task: Optional[asyncio.Task] = None
        
    def _build_ws_url(self) -> str:
        """Build Deepgram WebSocket URL with parameters."""
        params = [
            f"encoding=linear16",
            f"sample_rate={self.sample_rate}",
            f"channels=1",
            f"language={self.language}",
            f"model={self.model}",
            f"punctuate={str(self.punctuate).lower()}",
            f"smart_format={str(self.smart_format).lower()}",
            f"interim_results={str(self.interim_results).lower()}",
            "endpointing=300",  # 300ms silence for utterance end
        ]
        return f"wss://api.deepgram.com/v1/listen?{'&'.join(params)}"
        
    async def start(self) -> None:
        """Connect to Deepgram WebSocket API."""
        if self.is_running:
            logger.warning("DeepgramSTT already running")
            return
            
        try:
            logger.info(f"Connecting to Deepgram API (model: {self.model})...")
            
            # Connect with authentication header
            self.ws = await websockets.connect(
                self.ws_url,
                extra_headers={"Authorization": f"Token {self.api_key}"},
                ping_interval=5,
                ping_timeout=10
            )
            
            self.is_running = True
            self.connection_start_time = time.time()
            
            # Start receiving messages
            self.receive_task = asyncio.create_task(self._receive_loop())
            
            logger.info("Connected to Deepgram API")
            
        except Exception as e:
            logger.error(f"Failed to connect to Deepgram: {e}")
            raise
            
    async def stop(self) -> None:
        """Disconnect from Deepgram API."""
        if not self.is_running:
            return
            
        self.is_running = False
        
        try:
            # Send close message
            if self.ws and not self.ws.closed:
                await self.ws.send(json.dumps({"type": "CloseStream"}))
                await self.ws.close()
                
            # Cancel receive task
            if self.receive_task and not self.receive_task.done():
                self.receive_task.cancel()
                try:
                    await self.receive_task
                except asyncio.CancelledError:
                    pass
                    
            logger.info("Disconnected from Deepgram API")
            
        except Exception as e:
            logger.error(f"Error stopping Deepgram: {e}")
            
    async def add_audio(self, audio_chunk: np.ndarray) -> None:
        """Send audio chunk to Deepgram."""
        if not self.is_running or not self.ws or self.ws.closed:
            logger.warning("Cannot send audio: not connected to Deepgram")
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
            
            # Send to Deepgram
            await self.ws.send(audio_bytes)
            self.total_audio_sent += len(audio_bytes)
            
        except Exception as e:
            logger.error(f"Error sending audio to Deepgram: {e}")
            
    async def _receive_loop(self) -> None:
        """Receive and process messages from Deepgram."""
        try:
            async for message in self.ws:
                if not self.is_running:
                    break
                    
                try:
                    data = json.loads(message)
                    await self._handle_message(data)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse Deepgram message: {e}")
                except Exception as e:
                    logger.error(f"Error handling Deepgram message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("Deepgram connection closed")
        except Exception as e:
            if self.is_running:
                logger.error(f"Error in Deepgram receive loop: {e}")
                
    async def _handle_message(self, data: dict) -> None:
        """Handle a message from Deepgram."""
        msg_type = data.get("type")
        
        if msg_type == "Results":
            await self._handle_results(data)
        elif msg_type == "Metadata":
            logger.debug(f"Deepgram metadata: {data}")
        elif msg_type == "UtteranceEnd":
            logger.debug("Deepgram utterance end")
        elif msg_type == "SpeechStarted":
            logger.debug("Deepgram speech started")
        else:
            logger.debug(f"Unknown Deepgram message type: {msg_type}")
            
    async def _handle_results(self, data: dict) -> None:
        """Handle transcription results from Deepgram."""
        try:
            channel = data.get("channel", {})
            alternatives = channel.get("alternatives", [])
            
            if not alternatives:
                return
                
            # Get the best alternative
            alternative = alternatives[0]
            transcript = alternative.get("transcript", "").strip()
            confidence = alternative.get("confidence", 0.0)
            
            if not transcript:
                return
                
            # Determine if this is a final result
            is_final = data.get("is_final", False)
            speech_final = data.get("speech_final", False)
            
            # Calculate latency
            duration = data.get("duration", 0)
            start_time = data.get("start", 0)
            latency = time.time() - self.connection_start_time - start_time if self.connection_start_time else 0
            self.latencies.append(latency)
            
            # Create result
            result = STTResult(
                text=transcript,
                is_final=is_final or speech_final,
                confidence=confidence,
                timestamp=time.time()
            )
            
            if result.is_final:
                # Final result
                self.utterance_count += 1
                result.utterance_num = self.utterance_count
                self.last_final_text = transcript
                self.total_finals += 1
                
                logger.info(f"[FINAL #{self.utterance_count}] {transcript} (confidence: {confidence:.2f}, latency: {latency*1000:.0f}ms)")
                self._emit_final(result)
            else:
                # Partial result
                self.total_partials += 1
                logger.debug(f"[PARTIAL] {transcript} (confidence: {confidence:.2f}, latency: {latency*1000:.0f}ms)")
                self._emit_partial(result)
                
        except Exception as e:
            logger.error(f"Error processing Deepgram results: {e}")
            
    def get_stats(self) -> dict:
        """Get statistics about the Deepgram connection."""
        avg_latency = sum(self.latencies) / len(self.latencies) if self.latencies else 0
        
        return {
            "provider": "deepgram",
            "model": self.model,
            "is_running": self.is_running,
            "utterance_count": self.utterance_count,
            "total_audio_sent_bytes": self.total_audio_sent,
            "total_partials": self.total_partials,
            "total_finals": self.total_finals,
            "average_latency_ms": avg_latency * 1000,
            "connection_duration_seconds": time.time() - self.connection_start_time if self.connection_start_time else 0
        }
