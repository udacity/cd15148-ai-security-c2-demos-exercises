# Demo: Build a Prompt Injection Attack Against a Basic Chat Assistant

Estimated time: 14 minutes

## Overview

This demo shows how a basic chat assistant can be compromised by prompt injection. Learners test how malicious text embedded in a user message or prior conversation context can override intended assistant behavior.

The demo focuses on instruction handling rather than application code vulnerabilities. Students compare baseline chat behavior, direct prompt injection, conversation-history injection, and simple defensive controls.

## Scenario

A simple internal chat assistant answers employee questions. The application sends:

- A system prompt with safety and role instructions.
- Optional conversation history.
- A user prompt describing the request.

During a security review, the AI team tests whether malicious instructions in user-controlled content can manipulate the assistant.

## What Students Learn

- Prompt injection targets instruction parsing, not traditional code bugs.
- Conversation history and retrieved context must be treated as untrusted input.
- Direct injection is usually easier to spot than subtle context manipulation.
- System prompts alone are not sufficient.
- Output validation and instruction hierarchy checks reduce risk.

## Run the Demo

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Open:

```text
notebooks/prompt_injection_chat_assistant_demo.ipynb
```

The notebook runs in deterministic mock mode by default, so no API key is required.

## Optional OpenAI API Mode

To run against a live OpenAI model, set:

```bash
export OPENAI_API_KEY="..."
```

Then change `USE_OPENAI = False` to `USE_OPENAI = True` in the notebook.

The live API path uses the OpenAI Responses API with `client.responses.create(...)`, where the system instructions are passed through the `instructions` field and chat messages are passed as `input`.
