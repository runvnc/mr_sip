#!/usr/bin/env python3
"""
SIP Session Manager for MindRoot

Manages SIP sessions and their association with MindRoot conversation contexts.
Handles audio routing between SIP calls and MindRoot's TTS/STT systems.
"""

import asyncio
import threading
import logging
from typing import Dict, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class SIPSession:
    """
    Represents an active SIP call session linked to a MindRoot conversation.
    """
    
    def __init__(self, log_id: str, destination: str, baresip_bot=None):
        self.log_id = log_id
        self.destination = destination
        self.baresip_bot = baresip_bot
        self.created_at = datetime.now()
        self.is_active = False
        self.audio_queue = asyncio.Queue()
        self._audio_sender_task = None
        
    async def start_audio_sender(self):
        """Start the audio sender task for TTS output"""
        if self._audio_sender_task is None:
            self._audio_sender_task = asyncio.create_task(self._audio_sender_loop())
            
    async def stop_audio_sender(self):
        """Stop the audio sender task"""
        if self._audio_sender_task:
            self._audio_sender_task.cancel()
            try:
                await self._audio_sender_task
            except asyncio.CancelledError:
                pass
            self._audio_sender_task = None
            
    async def _audio_sender_loop(self):
        """Background task that sends audio chunks to the SIP call"""
        try:
            while self.is_active:
                try:
                    audio_chunk = await asyncio.wait_for(self.audio_queue.get(), timeout=1.0)
                    if audio_chunk is None:  # Sentinel to stop
                        break
                    await self._send_audio_to_sip(audio_chunk)
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Error in audio sender loop: {e}")
                    break
        except asyncio.CancelledError:
            logger.info(f"Audio sender cancelled for session {self.log_id}")
            
    async def _send_audio_to_sip(self, audio_chunk: bytes):
        """Send audio chunk to the SIP call"""
        # TODO: Implement actual audio sending to baresip
        # This will need to interface with baresip's audio output system
        if self.baresip_bot and hasattr(self.baresip_bot, 'send_audio'):
            try:
                await self.baresip_bot.send_audio(audio_chunk)
            except Exception as e:
                logger.error(f"Failed to send audio to SIP: {e}")
        else:
            logger.warning(f"No audio output method available for session {self.log_id}")
            
    async def send_audio(self, audio_chunk: bytes):
        """Queue audio chunk for sending to SIP call"""
        if self.is_active:
            try:
                await self.audio_queue.put(audio_chunk)
            except Exception as e:
                logger.error(f"Failed to queue audio chunk: {e}")
                
    async def end_session(self):
        """End the SIP session and cleanup resources"""
        logger.info(f"Ending SIP session {self.log_id}")
        self.is_active = False
        
        # Signal audio sender to stop
        try:
            await self.audio_queue.put(None)
        except:
            pass
            
        await self.stop_audio_sender()
        
        # Hangup the call if still active
        if self.baresip_bot and hasattr(self.baresip_bot, 'hang'):
            try:
                self.baresip_bot.hang()
            except Exception as e:
                logger.error(f"Error hanging up call: {e}")

class SIPSessionManager:
    """
    Manages multiple SIP sessions and their association with MindRoot contexts.
    """
    
    def __init__(self):
        self.sessions: Dict[str, SIPSession] = {}
        self._lock = asyncio.Lock()
        
    async def create_session(self, log_id: str, destination: str, baresip_bot=None) -> SIPSession:
        """Create a new SIP session"""
        async with self._lock:
            if log_id in self.sessions:
                logger.warning(f"Session {log_id} already exists, ending previous session")
                await self.end_session(log_id)
                
            session = SIPSession(log_id, destination, baresip_bot)
            self.sessions[log_id] = session
            logger.info(f"Created SIP session {log_id} for destination {destination}")
            return session
            
    async def get_session(self, log_id: str) -> Optional[SIPSession]:
        """Get an existing SIP session"""
        async with self._lock:
            return self.sessions.get(log_id)
            
    async def end_session(self, log_id: str) -> bool:
        """End a SIP session"""
        async with self._lock:
            session = self.sessions.get(log_id)
            if session:
                await session.end_session()
                del self.sessions[log_id]
                logger.info(f"Ended SIP session {log_id}")
                return True
            return False
            
    async def get_active_sessions(self) -> Dict[str, SIPSession]:
        """Get all active sessions"""
        async with self._lock:
            return {log_id: session for log_id, session in self.sessions.items() if session.is_active}
            
    async def cleanup_all_sessions(self):
        """Cleanup all sessions (called on shutdown)"""
        async with self._lock:
            for log_id in list(self.sessions.keys()):
                await self.end_session(log_id)
            logger.info("All SIP sessions cleaned up")

# Global session manager instance
_session_manager = None

def get_session_manager() -> SIPSessionManager:
    """Get or create the global SIP session manager"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SIPSessionManager()
    return _session_manager
