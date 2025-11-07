"""
Simple AI Voice Agent Example with PySIP
Focused on OpenAI Realtime API integration
"""
import asyncio
import audioop
from typing import Optional
from PySIP import SipCall
from PySIP.filters import CallState


class SimpleAIVoiceAgent:
    """
    Simple wrapper for AI voice agent using PySIP.
    Provides easy access to incoming/outgoing audio streams.
    """
    
    def __init__(self, sip_username: str, sip_password: str, sip_server: str):
        self.sip_username = sip_username
        self.sip_password = sip_password
        self.sip_server = sip_server
        
        self.call: Optional[SipCall] = None
        self.incoming_audio_queue = asyncio.Queue()
        self.is_active = False
        
        # Callbacks that you can override
        self.on_audio_received = None  # async def(audio_bytes: bytes)
        self.on_call_started = None    # async def()
        self.on_call_ended = None      # async def()
    
    async def make_call(self, phone_number: str):
        """Make an outbound call"""
        self.call = SipCall(
            username=self.sip_username,
            password=self.sip_password,
            route=self.sip_server,
            callee=phone_number
        )
        
        # Set up call state handler
        @self.call.on_call_state_changed
        async def on_state_changed(state):
            if state == CallState.ANSWERED:
                await self._on_call_answered()
            elif state in [CallState.ENDED, CallState.FAILED, CallState.BUSY]:
                await self._on_call_ended()
        
        # Set up audio frame handler
        @self.call.on_frame_received
        async def on_frame(frame: bytes):
            await self._on_audio_frame(frame)
        
        # Start the call
        print(f"Calling {phone_number}...")
        await self.call.start()
    
    async def _on_call_answered(self):
        """Called when call is answered"""
        print("Call answered!")
        self.is_active = True
        
        # Register our queue for incoming audio
        if self.call._rtp_session:
            self.call._rtp_session._output_queues['ai_agent'] = self.incoming_audio_queue
        
        # Start audio processing
        asyncio.create_task(self._process_audio_loop())
        
        # Call user callback
        if self.on_call_started:
            await self.on_call_started()
    
    async def _on_call_ended(self):
        """Called when call ends"""
        print("Call ended")
        self.is_active = False
        
        # Signal end of audio stream
        await self.incoming_audio_queue.put(None)
        
        # Call user callback
        if self.on_call_ended:
            await self.on_call_ended()
    
    async def _on_audio_frame(self, frame: bytes):
        """Called for each incoming audio frame"""
        await self.incoming_audio_queue.put(frame)
    
    async def _process_audio_loop(self):
        """Process incoming audio frames"""
        while self.is_active:
            try:
                frame = await asyncio.wait_for(
                    self.incoming_audio_queue.get(), 
                    timeout=1.0
                )
                
                if frame is None:
                    break
                
                # Call user's audio handler
                if self.on_audio_received:
                    await self.on_audio_received(frame)
                
            except asyncio.TimeoutError:
                continue
    
    async def send_audio(self, audio_bytes: bytes, format: str = 'pcm'):
        """
        Send audio to the call.
        
        Args:
            audio_bytes: Audio data
            format: 'pcm' (16-bit, 8kHz) or 'ulaw' (8-bit μ-law)
        """
        if not self.call or not self.is_active:
            print("Warning: No active call to send audio to")
            return
        
        # Convert to PCM if needed
        if format == 'ulaw':
            audio_bytes = audioop.ulaw2lin(audio_bytes, 2)
        
        # Send through call handler
        # Note: You'll need to implement this based on PySIP's API
        # This is a placeholder showing the concept
        # await self.call.call_handler.send_raw_audio(audio_bytes)
    
    async def hangup(self):
        """Hang up the call"""
        if self.call:
            await self.call.stop("Agent hangup")
    
    def pcm_to_ulaw(self, pcm_bytes: bytes) -> bytes:
        """Convert 16-bit PCM to 8-bit μ-law"""
        return audioop.lin2ulaw(pcm_bytes, 2)
    
    def ulaw_to_pcm(self, ulaw_bytes: bytes) -> bytes:
        """Convert 8-bit μ-law to 16-bit PCM"""
        return audioop.ulaw2lin(ulaw_bytes, 2)


# Example usage with OpenAI Realtime API (conceptual)
async def example_openai_realtime():
    """
    Example: Using the agent with OpenAI Realtime API
    """
    
    # Create agent
    agent = SimpleAIVoiceAgent(
        sip_username="your_username",
        sip_password="your_password",
        sip_server="sip.example.com:5060"
    )
    
    # Set up OpenAI client (pseudo-code)
    # openai_client = OpenAIRealtimeClient(api_key="your_key")
    
    # Handle incoming audio from call
    async def handle_call_audio(pcm_audio: bytes):
        """Send call audio to OpenAI"""
        # Convert to μ-law if OpenAI expects it
        ulaw_audio = agent.pcm_to_ulaw(pcm_audio)
        
        # Send to OpenAI
        # await openai_client.send_audio(ulaw_audio)
        
        print(f"Sent {len(pcm_audio)} bytes to OpenAI")
    
    # Handle audio from OpenAI
    # async def handle_openai_audio(audio_data: bytes):
    #     """Receive audio from OpenAI and send to call"""
    #     await agent.send_audio(audio_data, format='ulaw')
    
    # Set up callbacks
    agent.on_audio_received = handle_call_audio
    
    async def on_started():
        print("Call started - AI agent is now active")
        # Start OpenAI session
        # await openai_client.start_session()
    
    async def on_ended():
        print("Call ended - cleaning up")
        # End OpenAI session
        # await openai_client.end_session()
    
    agent.on_call_started = on_started
    agent.on_call_ended = on_ended
    
    # Make the call
    await agent.make_call("+1234567890")


# Example usage with STT -> LLM -> TTS loop
async def example_stt_llm_tts():
    """
    Example: Traditional STT -> LLM -> TTS pipeline
    """
    
    agent = SimpleAIVoiceAgent(
        sip_username="your_username",
        sip_password="your_password",
        sip_server="sip.example.com:5060"
    )
    
    # Audio buffer for STT
    audio_buffer = bytearray()
    buffer_duration_ms = 1000  # Process every 1 second
    buffer_size = int(8000 * 2 * buffer_duration_ms / 1000)  # 8kHz * 2 bytes * duration
    
    async def handle_call_audio(pcm_audio: bytes):
        """Buffer audio and send to STT when ready"""
        nonlocal audio_buffer
        
        audio_buffer.extend(pcm_audio)
        
        # When buffer is full, process it
        if len(audio_buffer) >= buffer_size:
            chunk = bytes(audio_buffer[:buffer_size])
            audio_buffer = audio_buffer[buffer_size:]
            
            # Send to STT
            # text = await stt_service.transcribe(chunk)
            # print(f"User said: {text}")
            
            # Send to LLM
            # response = await llm_service.generate(text)
            # print(f"AI response: {response}")
            
            # Convert to speech
            # audio = await tts_service.synthesize(response)
            
            # Send to call
            # await agent.send_audio(audio, format='pcm')
            
            print(f"Processed {len(chunk)} bytes of audio")
    
    agent.on_audio_received = handle_call_audio
    
    # Make the call
    await agent.make_call("+1234567890")


if __name__ == "__main__":
    print("Simple AI Voice Agent Example")
    print("=" * 50)
    print()
    print("This example shows how to:")
    print("1. Access incoming audio from a SIP call")
    print("2. Send audio back to the call")
    print("3. Integrate with AI services (OpenAI, STT/TTS, etc.)")
    print()
    print("Audio format: 16-bit PCM, 8kHz, mono")
    print("Codec: G.711 μ-law (PCMU)")
    print()
    print("To run:")
    print("  asyncio.run(example_openai_realtime())")
    print("  asyncio.run(example_stt_llm_tts())")
