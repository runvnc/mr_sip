API Reference
Speech to Text
Turn-based Audio (Flux)


Copy page

WSS

wss://api.deepgram.com
/v2/listen
Handshake
URL	wss://api.deepgram.com/v2/listen
Method	GET
Status	101 Switching Protocols
Try it
Messages

"0000FF00000000FF00000000010101010101010100000000FFFFFFFFFFFEFEFDFEFEFEFEFDFDFEFEFEFEFEFEFEFEFEFFFFFFFFFEFEFEFEFF0001000001020303030303030303030201010000FFFFFEFDFDFDFDFEFFFFFFFF0001020303020201000000FFFDFCFBFAFAFBFAF9F8F7F7F7F6F6F4F2F2F3F7FC000406090F14191A19181715110E0A05FEF9F6F3F0EEECEBEBECEEF2F6F9FC0005090D0F101010100E0C08041"
ListenV2Media


{ "type": "CloseStream" }
ListenV2CloseStream


{ "type": "Connected", "request_id": "550e8400-e29b-41d4-a716-446655440000", "sequence_id": 0 }
ListenV2Connected


{ "type": "TurnInfo", "request_id": "ad12514a-0d38-4f7e-8fba-cce10d8f174c", "sequence_id": 11, "event": "EndOfTurn", "turn_index": 0, "audio_window_start": 0, "audio_window_end": 1.3, "transcript": "Hello, how are you?", "words": [ { "word": "Hello,", "confidence": 0.96 }, { "word": "how", "confidence": 0.94 }, { "word": "are", "confidence": 0.97 }, { "word": "you?", "confidence": 0.92 } ], "end_of_turn_confidence": 0.86 }
ListenV2TurnInfo


{ "type": "Error", "sequence_id": 5, "code": "INTERNAL_SERVER_ERROR", "description": "An internal server error occurred while processing the request" }
ListenV2FatalError

Real-time conversational speech recognition with contextual turn detection for natural voice conversations

Handshake
WSS

wss://api.deepgram.com
/v2/listen

Headers
Authorization
string
Required
Use your API key for authentication, or alternatively generate a temporary token and pass it via the token query parameter.

Example: token %DEEPGRAM_API_KEY% or bearer %DEEPGRAM_TOKEN%

Query parameters
model
enum
Required
Defines the AI model used to process submitted audio.
Allowed values:
flux-general-en
encoding
enum
Required
Defaults to linear16
Encoding of the audio stream. Currently only supports raw signed little-endian 16-bit PCM.

Allowed values:
linear16
sample_rate
any
Required
Sample rate of the audio stream in Hz.
eager_eot_threshold
any
Optional
End-of-turn confidence required to fire an eager end-of-turn event. When set, enables EagerEndOfTurn and TurnResumed events. Valid Values 0.3 - 0.9.

eot_threshold
any
Optional
End-of-turn confidence required to finish a turn. Valid Values 0.5 - 0.9.

eot_timeout_ms
any
Optional
A turn will be finished when this much time has passed after speech, regardless of EOT confidence.
keyterm
string or list of strings
Optional
Keyterm prompting can improve recognition of specialized terminology. Pass multiple keyterm query parameters to boost multiple keyterms.

Show 2 variants
mip_opt_out
any
Optional
Opts out requests from the Deepgram Model Improvement Program. Refer to our Docs for pricing impacts before setting this to true. https://dpgr.am/deepgram-mip

tag
any
Optional
Label your requests for the purpose of identification during usage reporting
Send
ListenV2Media
any
Required
Send audio or video data to be transcribed
OR
ListenV2CloseStream
any
Required
Send a CloseStream message to close the WebSocket stream
Receive
ListenV2Connected
any
Required
Receive a connected message
OR
ListenV2TurnInfo
any
Required
Receive a turn info message
OR
ListenV2FatalError
any
Required
Receive a fatal error message
Was this page helpful?
Yes

