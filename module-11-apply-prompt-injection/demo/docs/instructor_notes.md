# Instructor Notes

This demo is intentionally safe and deterministic by default. The mock assistant simulates common prompt-injection failure modes so the classroom result does not depend on live model variability or API availability.

## Suggested Flow

1. Run one benign baseline query.
2. Show the system prompt and chat message assembly.
3. Run direct injection payloads in the user prompt.
4. Run context/history injection payloads.
5. Compare unsafe orchestration with guarded orchestration.
6. Emphasize that the vulnerable surface is instruction handling, not a normal code injection bug.

## Talking Points

- The same benign user query can produce different output when attacker-controlled text is nearby.
- Conversation history is not inherently trustworthy.
- System prompts are necessary but not sufficient.
- Output validation can catch obvious compromise markers, but it is not a complete defense.
- Stronger designs isolate untrusted content and constrain what the model is allowed to do with it.
