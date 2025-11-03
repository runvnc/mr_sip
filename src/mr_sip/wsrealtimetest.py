import base64
import json
import struct
import soundfile as sf
import traceback
# for playing local audio
import pyaudio
import wave

import os
import websocket
import simpleaudio as sa

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

url = "wss://api.openai.com/v1/realtime?model=gpt-realtime"
headers = ["Authorization: Bearer " + OPENAI_API_KEY]


sys="""
# Persona

Your name is Erica. 

You speak American English.

# Instructions

Say hello, give them your name, then output the tone for digit '5'.
Ignore anything the user says for now.

# Output channel tool

Use the output() function with the text argument to output non-verbal content.

# JSON commands

Example:

"send_dtmf": {"digits": "1234"}

"""

instr1 = {
  "type": "session.update",
  "session": {
    "type": "realtime",
    "instructions": sys,
    "tools": [
        {
            "type": "function",
            "name": "output",
            "description": "Call this function for all non-verbal outputs. You can output text or JSON-encoded function calls.",
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
    "tool_choice": "auto",
  },
  "event_id": "5fc543c4-f59c-420f-8fb9-68c45d1546a7a2"
}

instr2 = {
  "type": "session.update",
  "session": {
    "type": "realtime",
    "instructions": "Your name is Erica. If they give you a name, immediately output that name on your text channel. Leave a voicemail asking them to call you back.",
  },
  "event_id": "5fc543c4-f59c-420f-8fb9-68c45d1546a7a",
}

files = [
    '/files/upd6/mr_verification_dashboard/audio/voicemailpadded.wav'
]

def send_wavs():
    try:
        print("Top of send_wavs")
        for filename in files:
            data, samplerate = sf.read(filename, dtype='float32')
            channel_data = data[:, 0] if data.ndim > 1 else data
            base64_chunk = base64_encode_audio(channel_data)

            # Send the client event
            event = {
                "type": "input_audio_buffer.append",
                "audio": base64_chunk
            }
            print("Sending audio data")
            ws.send(json.dumps(event))
            print('sent audio chunk')
    except Exception as e:
        print(e)

def on_open(ws):
    print("Connected to server.")
    ws.send(json.dumps(instr1))
    print("Sent instructions")
    send_wavs()
    print("Sent wavs")

def float_to_16bit_pcm(float32_array):
    clipped = [max(-1.0, min(1.0, x)) for x in float32_array]
    pcm16 = b''.join(struct.pack('<h', int(x * 32767)) for x in clipped)
    return pcm16

def base64_encode_audio(float32_array):
    pcm_bytes = float_to_16bit_pcm(float32_array)
    encoded = base64.b64encode(pcm_bytes).decode('ascii')
    return encoded

p = None
stream = None

def setup_audio_output_stream():
    global stream
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    output=True)

def on_message(ws, message):
    global stream
    try:
        server_event = json.loads(message)
        if server_event['type'] == "response.output_audio.delta":
            print("received audio")
            audio_bytes = base64.b64decode(server_event['delta'])
            play_obj = sa.play_buffer(audio_bytes, 1, 2, 24000)
            play_obj.wait_done()
            print("played audio?")
            #stream.write(audio_bytes)
        elif server_event['type'] == "conversation.item.done":
            item = server_event['item']
            if item['type'] == "function_call" and item['name'] == "output":
                arguments = json.loads(item['arguments'])
                print("Function call output:")
                print(arguments['text'])
                try:
                    cmd = json.loads(arguments['text'])
                    print("Parsed command:")
                    print(cmd)
                except json.JSONDecodeError:
                    pass
                    #print("Not a JSON command.")
        else:
            print("received message:")
            print(message)

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

#setup_audio_output_stream()

ws.run_forever()


