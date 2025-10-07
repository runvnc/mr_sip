---
title: Optimize Voice Agent Latency with Eager End of Turn
subtitle: >-
  Reduce end-to-end latency by preparing responses early with Eager End of Turn
  events.
slug: docs/flux/voice-agent-eager-eot
---

Eager end of turn processing is the practice of starting LLM processing on **medium-confidence transcripts** (`EagerEndOfTurn` events) before waiting for a high-confidence `EndOfTurn`. By overlapping LLM generation with user speech, you can cut **hundreds of milliseconds** from your agent's response time.

<Info title="Important Note">
  `EagerEndOfTurn` and `TurnResumed` events are ONLY triggered if you have configured the `eager_eot_threshold` in your connection string.
</Info>

## How Eager End of Turn Processing Works

1. **Receive `EagerEndOfTurn`**
   - Flux is moderately confident the user has finished speaking.
   - Send the transcript downstream to your LLM and begin preparing a reply.

2. **If `TurnResumed` occurs**
   - The user wasn't finished after all.
   - Cancel the in-progress response and wait for the next `EagerEndOfTurn` or `EndOfTurn`.

3. **If `EndOfTurn` occurs**
   - The user is done speaking with high confidence.
   - Finalize and deliver the response you've already started preparing.
   - `EndOfTurn` transcript will exactly match the `EagerEndOfTurn` transcript, ensuring consistent transcription throughout the turn lifecycle.

## Implementation Strategies

### 1. Simple (Recommended Starting Point)
- Use **only `EndOfTurn`** events.
- Simplest implementation and minimal LLM calls
- Ideal for majority of developers

### 2. Optimized (With Eager End of Turn Processing)
- Use **both `EagerEndOfTurn` and `EndOfTurn`** events.
- Reduce latency by preparing replies early with speculative response generation.
- Expect more LLM calls and slightly more complexity.
- Recommended once you’re confident in your pipeline and want production-grade performance.

## Tips & Tricks for Eager End of Turn Processing

### Tune Confidence Thresholds
- `eager_eot_threshold`: Lower values → earlier triggers, but more false starts.
- `eot_threshold`: Higher values → more reliable `EndOfTurn`, but may increase latency.
- Experiment with values to balance **speed vs. stability**.

### Handle `TurnResumed` Gracefully
- Treat `TurnResumed` as a cancellation signal.
- Be ready to discard or revise any LLM replies in progress.
- Consider a retry strategy if this happens often in your use case.

### Keep Responses Flexible
- Avoid committing to a reply until `EndOfTurn`.
- Use `EagerEndOfTurn` outputs to **draft**, not finalize.
- Build resilience for handling `TurnResumed` events when the user continues speaking.

### Optimize LLM Cost
- Eager end of turn processing means **more LLM requests**.
- To reduce spend:
  - Use smaller/faster models for EagerEndOfTurn drafts.
  - Only call the full LLM on `EndOfTurn`.
  - Cache prepared responses and reuse on `EndOfTurn` (transcript guaranteed to match).

### Monitor and Log Events
- Track how often `EagerEndOfTurn` → `TurnResumed` vs. `EagerEndOfTurn` → `EndOfTurn`.
- Use this data to refine thresholds and tune your pipeline.

## Example Code

This code demonstrates how to handle Flux message events to implement eager end-of-turn processing. The examples show message parsing and event handling for the three critical events: -

- `EagerEndOfTurn` (start preparing response)
- `TurnResumed` (cancel draft response)
- `EndOfTurn` (finalize and deliver response)


<CodeGroup>
```javascript
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type !== 'TurnInfo') return;

  switch (data.event) {
    case 'EagerEndOfTurn':
      console.log('EagerEndOfTurn:', data.transcript);
      prepareDraftResponse(data.transcript);
      break;

    case 'TurnResumed':
      console.log('User kept speaking, cancel draft');
      cancelDraftResponse();
      break;

    case 'EndOfTurn':
      console.log('Final:', data.transcript);
      finalizeResponse(data.transcript);
      break;
  }
};
```
```Python
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
```
``` csharp C#
public void OnMessage(FluxMessage message)
{
    // Parse incoming Flux message
    if (message.Type != "TurnInfo") return;

    switch (message.Event)
    {
        case "EagerEndOfTurn":
            Console.WriteLine($"EagerEndOfTurn: {message.Transcript}");
            PrepareDraftResponse(message.Transcript);
            break;

        case "TurnResumed":
            Console.WriteLine("User kept speaking, cancel draft");
            CancelDraftResponse();
            break;

        case "EndOfTurn":
            Console.WriteLine($"Final: {message.Transcript}");
            FinalizeResponse(message.Transcript);
            break;
    }
}

private void PrepareDraftResponse(string transcript)
{
    // Start preparing LLM response with moderate confidence transcript
}

private void CancelDraftResponse()
{
    // Cancel any in-progress response preparation
}

private void FinalizeResponse(string transcript)
{
    // Use the final transcript to deliver the response
}
```
``` Go
func onMessage(message *FluxMessage) {
    // Parse incoming Flux message
    if message.Type != "TurnInfo" {
        return
    }

    switch message.Event {
    case "EagerEndOfTurn":
        fmt.Printf("EagerEndOfTurn: %s\n", message.Transcript)
        prepareDraftResponse(message.Transcript)

    case "TurnResumed":
        fmt.Println("User kept speaking, cancel draft")
        cancelDraftResponse()

    case "EndOfTurn":
        fmt.Printf("Final: %s\n", message.Transcript)
        finalizeResponse(message.Transcript)
    }
}

func prepareDraftResponse(transcript string) {
    // Start preparing LLM response with moderate confidence transcript
}

func cancelDraftResponse() {
    // Cancel any in-progress response preparation
}

func finalizeResponse(transcript string) {
    // Use the final transcript to deliver the response
}
```

</CodeGroup>

## Summary

### When to Use Eager End of Turn

- High-interruption environments (e.g., call centers, IVRs).
- Conversational agents where natural back-and-forth timing matters.
- Latency-sensitive apps where response speed is critical to user experience.
- If your LLM configuration is complex and has high latency issues. e.g., Good for trimming that last 100-200ms of end-to-end latency at the cost of 50-70% more LLM calls.
- If Your LLM configuration has complex RAG (Retrieval-Augmented Generation) or Function Calling involved.

### When to use End of Turn only

- Most developers will find `EndOfTurn` detection sufficiently fast enough to support natural conversation, but not all voice AI workflows are the same.
- For more complex and expensive voice AI workflows, it might be worthwhile to use `EagerEndOfTurn`to call LLMs speculatively, i.e., in preparation for an upcoming turn end, in order to minimize response latency.

