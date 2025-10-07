async def main():
    """Main async function to handle URL streaming to Deepgram Flux"""

    # Create the Deepgram async client
    client = AsyncDeepgramClient() # The API key retrieval happens automatically in the constructor

    try:
        # Connect to Flux with auto-detection for streaming audio
        async with client.listen.v2.connect(
            model="flux-general-en",
            encoding="linear16",
            sample_rate="16000"
        ) as connection:

            # Define message handler function
            def on_message(message: ListenV2SocketClientResponse) -> None:
                msg_type = getattr(message, "type", "Unknown")

                # Show transcription results
                if hasattr(message, 'transcript') and message.transcript:
                    print(f"🎤 {message.transcript}")

                    # Show word-level confidence with color coding
                    if hasattr(message, 'words') and message.words:
                        colored_words = []
                        for word in message.words:
                            color = get_confidence_color(word.confidence)
                            colored_words.append(f"{color}{word.word}({word.confidence:.2f}){Colors.RESET}")
                        words_info = " | ".join(colored_words)
                        print(f"   📝 {words_info}")
                elif msg_type == "Connected":
                    print(f"✅ Connected to Deepgram Flux - Ready for audio!")

            # Set up event handlers
            connection.on(EventType.OPEN, lambda _: print("Connection opened"))
            connection.on(EventType.MESSAGE, on_message)
            connection.on(EventType.CLOSE, lambda _: print("Connection closed"))
            connection.on(EventType.ERROR, lambda error: print(f"Caught: {error}"))

            # Start the connection listening in background (it's already async)
            deepgram_task = asyncio.create_task(connection.start_listening())

            # Convert BBC stream to linear16 PCM using ffmpeg
            print(f"Starting to stream and convert audio from: {STREAM_URL}")

            # Use ffmpeg to convert the compressed BBC stream to linear16 PCM at 16kHz
            ffmpeg_cmd = [
                'ffmpeg',
                '-i', STREAM_URL,           # Input: BBC World Service stream
                '-f', 's16le',              # Output format: 16-bit little-endian PCM (linear16)
                '-ar', '16000',             # Sample rate: 16kHz
                '-ac', '1',                 # Channels: mono
                '-'                         # Output to stdout
            ]

            try:
                # Start ffmpeg process
                process = await asyncio.create_subprocess_exec(
                    *ffmpeg_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

                print(f"✅ Audio conversion started (BBC → linear16 PCM)")

                # Read converted PCM data and send to Deepgram
                while True:
                    chunk = await process.stdout.read(1024)
                    if not chunk:
                        break

                    # Send converted linear16 PCM data to Flux
                    await connection._send(chunk)

                await process.wait()

            except Exception as e:
                print(f"Error during audio conversion: {e}")
                if 'process' in locals():
                    stderr = await process.stderr.read()
                    print(f"FFmpeg error: {stderr.decode()}")

            # Wait for Deepgram task to complete (or cancel after timeout)
            try:
                await asyncio.wait_for(deepgram_task, timeout=60)
            except asyncio.TimeoutError:
                print("Stream timeout after 60 seconds")
                deepgram_task.cancel()

    except Exception as e:
        print(f"Caught: {e}")

if __name__ == "__main__":
    asyncio.run(main())


