# Demo: Build a Prompt Injection Attack Against a Basic Chat Assistant

Estimated time: 14 minutes

## Overview

This demo shows how a basic chat assistant can be compromised by prompt injection. Learners test how malicious text embedded in a user message or prior conversation context can influence live OpenAI model behavior.

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

Copy `.env.example` to `.env` or paste the classroom key into the notebook setup cell. Vocareum-issued keys that start with `voc-` automatically use `https://openai.vocareum.com/v1`.

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://openai.vocareum.com/v1"
```

The notebook uses `gpt-4.1-mini` by default and calls `client.chat.completions.create(...)` for both unsafe and guarded prompt runs.
