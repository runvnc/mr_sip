"""Protocol definitions for MindRoot SIP services.

This module defines typed interfaces for SIP telephony services.
Plugins and other code can import these Protocols for IDE autocomplete
and type checking.

Usage (recommended - use pre-instantiated proxy):
    from mr_sip import sip
    
    result = await sip.dial_service('555-1234', context=ctx)
    await sip.end_call_service(context=ctx)

Alternative (create your own proxy):
    from mr_sip.protocols import SIP
    from lib.providers.services import service_manager
    
    sip: SIP = service_manager.typed(SIP)
    result = await sip.dial_service('555-1234', context=ctx)
"""

from typing import Protocol, runtime_checkable, Any, Dict, Optional


@runtime_checkable
class SIP(Protocol):
    """SIP telephony service protocol.
    
    Provides voice call capabilities via SIP.
    Implemented by: mr_sip plugin
    """
    
    async def dial_service(
        self,
        destination: str,
        context: Any = None,
        enable_recording: bool = None,
        use_process_isolation: bool = True
    ) -> Dict[str, Any]:
        """Initiate a SIP call.
        
        Args:
            destination: Phone number or SIP URI to call
            context: MindRoot context (required for session linking)
            enable_recording: Override default recording setting
            use_process_isolation: Run PySIP in separate process
            
        Returns:
            Dict with status, log_id, destination, and other call info
        """
        ...
    
    async def end_call_service(
        self,
        context: Any = None
    ) -> Dict[str, Any]:
        """Terminate an active SIP call.
        
        Args:
            context: MindRoot context (required for session identification)
            
        Returns:
            Dict with status and call duration info
        """
        ...
    
    async def sip_audio_out_chunk(
        self,
        audio_chunk: bytes,
        timestamp: float = None,
        context: Any = None
    ) -> None:
        """Send audio chunk to active SIP call.
        
        Args:
            audio_chunk: Audio data as bytes (ulaw 8kHz)
            timestamp: When this audio should start playing
            context: MindRoot context
        """
        ...
    
    async def sip_clear_audio_queue(
        self,
        context: Any = None
    ) -> Dict[str, Any]:
        """Clear queued audio for interruption handling.
        
        Called when user interrupts to stop current response.
        
        Args:
            context: MindRoot context
            
        Returns:
            Dict with status info
        """
        ...


# Register with the protocol registry for discovery
try:
    from lib.providers.protocols import register_protocol
    register_protocol('sip', SIP)
except ImportError:
    # Protocol registry not available, that's fine
    pass


# Pre-instantiated lazy proxy for convenient access
# Usage: from mr_sip.protocols import sip
try:
    from lib.providers.protocols.registry import create_lazy_proxy
    sip: SIP = create_lazy_proxy(SIP)
except ImportError:
    # Protocols not available, sip proxy won't be available
    sip = None  # type: ignore
