def on_message(message):
    # Parse incoming Flux message
    if not hasattr(message, 'type') or message.type != 'TurnInfo':
        return

    event = getattr(message, 'event', None)
    transcript = getattr(message, 'transcript', '')

    if event == 'EagerEndOfTurn':
        print(f'EagerEndOfTurn: {transcript}')
        prepare_draft_response(transcript)

    elif event == 'TurnResumed':
        print('User kept speaking, cancel draft')
        cancel_draft_response()

    elif event == 'EndOfTurn':
        print(f'Final: {transcript}')
        finalize_response(transcript)

def prepare_draft_response(transcript):
    # Start preparing LLM response with moderate confidence transcript
    pass

def cancel_draft_response():
    # Cancel any in-progress response preparation
    pass

def finalize_response(transcript):
    # Use the final transcript to deliver the response
    pass

