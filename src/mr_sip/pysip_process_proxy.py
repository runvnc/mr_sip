#!/usr/bin/env python3
"""
PySIP Process Proxy - Main process interface to PySIP subprocess

This proxy runs in the main process and provides the same interface as
MindRootSIPBotS2S, but forwards all operations to the PySIP subprocess
via queues.

This allows the rest of the code to remain unchanged - it just uses the
proxy instead of the real bot.
"""

import asyncio
import logging
from typing import Optional
from datetime import datetime
from .pysip_process_wrapper import PySIPProcessWrapper

logger = logging.getLogger(__name__)

class PySIPProcessProxy:
    """Proxy for MindRootSIPBotS2S that runs in main process.
    
    This provides the same interface as MindRootSIPBotS2S but forwards
    all operations to the PySIP subprocess via the wrapper.
    
    The session manager and other code can use this exactly like the
    real bot - they won't know the difference.
    """
    
    def __init__(self, wrapper: PySIPProcessWrapper, context):
        """Initialize the proxy.
        
        Args:
            wrapper: PySIPProcessWrapper instance managing the subprocess
            context: MindRoot context
        """
        self.wrapper = wrapper
        self.context = context
        
        # State tracking (mirrors bot state)
        self.is_active = False
        self.call_established = False
        self.call_start_time: Optional[datetime] = None
        
        # Audio forwarding task
        self._audio_forwarder_task: Optional[asyncio.Task] = None
        
        logger.info(f"PySIP process proxy initialized for context {context.log_id}")
        
    async def make_call(self, destination: str, user: str, password: str, 
                       gateway: str, enable_recording: bool = False,
                       recording_dir: str = "recordings",
                       record_separate: bool = False):
        """Initiate call via subprocess.
        
        Args:
            destination: Phone number to call
            user: SIP username
            password: SIP password
            gateway: SIP gateway
            enable_recording: Enable call recording
            recording_dir: Directory for recordings
            record_separate: Save separate incoming/outgoing files
        """
        logger.info(f"Proxy: Making call to {destination}")
        
        # Start the subprocess and call
        success = await self.wrapper.start_call(
            user=user,
            password=password,
            gateway=gateway,
            destination=destination,
            enable_recording=enable_recording,
            recording_dir=recording_dir,
            record_separate=record_separate
        )
        
        if success:
            self.is_active = True
            self.call_established = True
            self.call_start_time = datetime.now()
            
            # Start audio forwarding from subprocess to OpenAI
            self._audio_forwarder_task = asyncio.create_task(
                self._forward_audio_to_openai()
            )
            
            logger.info(f"Proxy: Call established to {destination}")
        else:
            raise Exception("Failed to establish call")
            
    async def _forward_audio_to_openai(self):
        """Forward audio from subprocess to OpenAI.
        
        This task runs continuously, reading audio from the subprocess
        and sending it to OpenAI via the service manager.
        """
        from lib.providers.services import service_manager
        
        logger.info("Audio forwarder started")
        
        try:
            while self.is_active:
                # Get audio from subprocess
                audio_chunk = await self.wrapper.receive_audio()
                
                if audio_chunk:
                    # Send to OpenAI
                    try:
                        await service_manager.send_s2s_audio_chunk(
                            audio_bytes=audio_chunk,
                            context=self.context
                        )
                    except Exception as e:
                        logger.error(f"Error sending audio to OpenAI: {e}")
                else:
                    # No audio available, brief sleep
                    await asyncio.sleep(0.01)
                    
        except asyncio.CancelledError:
            logger.info("Audio forwarder cancelled")
        except Exception as e:
            logger.error(f"Error in audio forwarder: {e}")
        finally:
            logger.info("Audio forwarder exiting")
            
    async def send_tts_audio(self, audio_chunk: bytes):
        """Send TTS audio to subprocess (from OpenAI to phone).
        
        This is called by the session manager when OpenAI sends audio.
        
        Args:
            audio_chunk: Audio data from OpenAI (ulaw 8kHz)
        """
        if not self.is_active:
            logger.warning("Proxy: Cannot send audio - call not active")
            return
            
        await self.wrapper.send_audio(audio_chunk)
        
    def clear_audio_queue(self):
        """Clear all queued audio (for interruption)."""
        self.wrapper.clear_audio_queue()
        
    async def hangup_call(self):
        """Hangup the call."""
        logger.info("Proxy: Hangup requested")
        
        self.is_active = False
        self.call_established = False
        
        # Stop audio forwarder
        if self._audio_forwarder_task:
            self._audio_forwarder_task.cancel()
            try:
                await self._audio_forwarder_task
            except asyncio.CancelledError:
                pass
                
        # Stop subprocess
        await self.wrapper.stop()
        
        logger.info("Proxy: Call ended")
        
    def hang(self):
        """Synchronous hangup (for compatibility with session manager)."""
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_closed():
                asyncio.create_task(self.hangup_call())
                logger.info("Proxy: Async hangup scheduled")
        except Exception as e:
            logger.error(f"Proxy: Error scheduling hangup: {e}")
            
    def get_metrics(self):
        """Get metrics from subprocess.
        
        Returns:
            Dictionary with metrics
        """
        metrics = self.wrapper.get_metrics()
        metrics['proxy_active'] = self.is_active
        metrics['proxy_call_established'] = self.call_established
        return metrics
