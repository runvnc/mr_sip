import base64
import json
import struct
import soundfile as sf
import traceback
# for playing local audio
import pyaudio
import wave
import time
import asyncio
import os
import websocket
import simpleaudio as sa

openai_sockets = {}

@service()
async def start_s2s(model='gpt-realtime', system_prompt, on_command, on_audio_chunk=None, voice='marin',
                    play_local=False, context=None, **kwargs):
    """
        Start a speech-to-speech OpenAI realtime websocket session.
        Session will be identifiedby context.log_id

        Arguments:

            model: model name, e.g. 'gpt-realtime'

            system_prompt: system prompt string

            on_command: async callback function to handle function call commands from the server.
                        Arg 1: function name, arg 2: function parameters dict.

            on_audio_chunk: async callback function to handle audio chunks from the server.
                            Arg: audio bytes in float32 PCM format at 24000 Hz sample rate.

    """
    global openai_sockets
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

    url = "wss://api.openai.com/v1/realtime?model=gpt-realtime"
    headers = ["Authorization: Bearer " + OPENAI_API_KEY]


    def on_message(ws, message):
        try:
            server_event = json.loads(message)
            if server_event['type'] == "response.output_audio.delta":
                audio_bytes = base64.b64decode(server_event['delta'])
                await on_audio_chunk(audio_bytes, context=context)
                if play_local:
                    play_obj = sa.play_buffer(audio_bytes, 1, 2, 24000)
                    play_obj.wait_done()
            elif server_event['type'] == "conversation.item.done":
                item = server_event['item']
                if item['type'] == "function_call":
                    arguments = json.loads(item['arguments'])
                    try:
                        cmd = json.loads(arguments['text'])
                        await on_command(item['name'], cmd, context=context)
                    except json.JSONDecodeError:
                        pass
                    except Exception as e:
                        print("Error in on_command callback:")
                        trace = traceback.format_exc()
                        print(e)
                        print(trace)
                        raise e
            else:
                print("received message:")
                print(message)
            except Exception as e:
            trace = traceback.format_exc()
            print(e)
            print(trace)

    def on_open(ws):
        try: 
            print("OpenAI realtime websocket connected to server.")
            session_update = {
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "instructions": system_prompt,
                    "audio": {"output" : { "voice": voice} },
                    "tools": [
                    {
                        "type": "function",
                        "name": "output",
                        "description": "Call this function with JSON-encoded function calls if necessary.",
                        "parameters": {
                            "type": "object",
                            "strict": True,
                            "properties": {
                                "text": {
                                    "type": "string",
                                    "description": "Text, or properly escaped JSON for the command arguments"
                                }
                            }
                        }
                    }
                    ],
                    "tool_choice": "auto"
                }
            #"event_id": "5fc543c4-f59c-420f-8fb9-68c45d1546a7a2"
            }
            ws.send(json.dumps(session_update))
            print("OpenAI realtime initialized session.")
        except Exception as e:
            trace = traceback.format_exc()
            print(e)
            print(trace)

    ws = websocket.WebSocketApp(
        url,
        header=headers,
        on_open=on_open,
        on_message=on_message,
    )
    openai_sockets[context.log_id] = ws
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, ws.run_forever)


@service()
async def send_s2s_audio_chunk(audio_bytes, context=None):
    """
        Send an audio chunk to the server for processing.
        context.log_id identifies the session.

        audio_bytes: bytes of audio data in float 32 PCM format
                     at 24000 Hz sample rate.
    """
    global openai_sockets
    float32_array = struct.unpack('<' + 'f' * (len(audio_bytes) // 4), audio_bytes)
    base64_chunk = base64_encode_audio(float32_array)
    event = {
        "type": "input_audio_buffer.append",
        "audio": base64_chunk
    }
    ws = openai_sockets.get(context.log_id)
    if ws:
        ws.send(json.dumps(event))
    else:
        raise Exception(f"No active OpenAI socket for log_id {context.log_id}")

