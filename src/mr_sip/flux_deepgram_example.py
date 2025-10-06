import asyncio
import os
import sys
import json
import urllib.request

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

AUDIO_FILE = "audio/spacewalk_linear16.wav"  # Your audio file. Must be linear16.

async def main():
    """Main demo function."""
    print("🚀 Deepgram Flux Agent Demo")
    print("=" * 40)

    # Check for audio file
    if not os.path.exists(AUDIO_FILE):
        print(f"❌ Audio file '{AUDIO_FILE}' not found")
        print("Please add an audio.wav file to this directory")
        return

    # Read audio file
    print(f"📁 Reading {AUDIO_FILE}...")
    with open(AUDIO_FILE, 'rb') as f:
        audio_data = f.read()

    print(f"✓ Read {len(audio_data)} bytes")

    # Import Deepgram
    from deepgram import DeepgramClient
    from deepgram.core.events import EventType
    from deepgram.extensions.types.sockets import ListenV2SocketClientResponse, SpeakV1SocketClientResponse, SpeakV1ControlMessage, ListenV2MediaMessage, SpeakV1TextMessage

    client = DeepgramClient() # The API key retrieval happens automatically in the constructor

    # Transcribe with Flux
    print("\n🎤 Transcribing with Flux...")
    transcript = ""
    done = asyncio.Event()

    def on_flux_message(message: ListenV2SocketClientResponse) -> None:
        nonlocal transcript
        if hasattr(message, 'type') and message.type == 'TurnInfo':
            if hasattr(message, 'event') and message.event == 'EndOfTurn':
                if hasattr(message, 'transcript') and message.transcript:
                    transcript = message.transcript.strip()
                    print(f"✓ Transcript: '{transcript}'")
                    done.set()

    with client.listen.v2.connect(model="flux-general-en", encoding="linear16", sample_rate=16000) as connection:
        connection.on(EventType.MESSAGE, on_flux_message)

        import threading
        threading.Thread(target=connection.start_listening, daemon=True).start()

        # Send audio in chunks
        chunk_size = 4096
        for i in range(0, len(audio_data), chunk_size):
            connection.send_media(audio_data[i:i + chunk_size])
            await asyncio.sleep(0.01)

        # Wait for transcript
        await asyncio.wait_for(done.wait(), timeout=30.0)

    if not transcript:
        print("❌ No transcript received")
        return

    # Generate OpenAI response
    print("\n🤖 Generating OpenAI response...")

    # Direct HTTP request to OpenAI API
    openai_data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Keep responses concise and conversational."},
            {"role": "user", "content": transcript}
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }

    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(openai_data).encode(),
        headers={
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
            "Content-Type": "application/json"
        }
    )

    try:
        with urllib.request.urlopen(req) as response_obj:
            openai_response = json.loads(response_obj.read().decode())
            response = openai_response["choices"][0]["message"]["content"]
            print(f"✓ Response: '{response}'")
    except Exception as e:
        print(f"❌ OpenAI API error: {e}")
        response = f"I heard you say: {transcript}"  # Fallback
        print(f"✓ Fallback response: '{response}'")

    # Generate TTS Response
    print("\n🔊 Generating TTS...")
    tts_audio = []
    tts_done = asyncio.Event()

    def on_tts_message(message: SpeakV1SocketClientResponse) -> None:
        if isinstance(message, bytes):
            tts_audio.append(message)
        elif hasattr(message, 'type') and message.type == 'Flushed':
            tts_done.set()

    with client.speak.v1.connect(model="aura-2-phoebe-en", encoding="linear16", sample_rate=16000) as connection:
        connection.on(EventType.MESSAGE, on_tts_message)

        threading.Thread(target=connection.start_listening, daemon=True).start()

        connection.send_text(SpeakV1TextMessage(type="Speak", text=response))
        connection.send_control(SpeakV1ControlMessage(type="Flush"))

        # Wait for TTS completion
        await asyncio.wait_for(tts_done.wait(), timeout=15.0)

    # Save TTS audio
    if tts_audio:
        output_file = "audio/responses/agent_response.wav"
        combined_audio = b''.join(tts_audio)

        # Create simple WAV header
        import struct
        wav_header = struct.pack(
            '<4sI4s4sIHHIIHH4sI',
            b'RIFF', 36 + len(combined_audio), b'WAVE', b'fmt ', 16, 1, 1,
            16000, 32000, 2, 16, b'data', len(combined_audio)
        )

        with open(output_file, 'wb') as f:
            f.write(wav_header + combined_audio)

        print(f"💾 Saved TTS audio: {output_file}")

    print("\n🎉 Demo complete!")
    print(f"📝 User: '{transcript}'")
    print(f"🤖 Agent: '{response}'")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Demo stopped")
    except Exception as e:
        print(f"❌ Error: {e}")

