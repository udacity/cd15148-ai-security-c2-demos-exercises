# Instructor Notes

This demo uses a live OpenAI model by default. Learners should use the Vocareum OpenAI key from the classroom Cloud Resources tab; keys that start with `voc-` are routed to `https://openai.vocareum.com/v1`.

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
- Output validation should check semantic indicators of unsafe behavior, not hardcoded success labels.
- Stronger designs isolate untrusted content and constrain what the model is allowed to do with it.
- The demo deliberately uses `gpt-4.1-mini` because the audit evidence showed observable attack behavior on this weaker target. Frontier models such as `gpt-4o` and `gpt-5-mini` resisted the same earlier payload set more often, so the lesson is not that every modern model fails by default.
