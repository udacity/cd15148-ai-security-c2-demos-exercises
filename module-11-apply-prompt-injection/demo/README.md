# Demo: Build a Prompt Injection Attack Against a Basic Chat Assistant

Estimated time: 14 minutes

## Overview

This demo shows how a basic chat assistant can be compromised by prompt injection. Learners test how malicious text embedded in a user message or prior conversation context can influence live model behavior.

The notebook runs against local [Ollama](https://ollama.com) models, so it needs no API key and runs on a laptop. It compares six small models side by side, each under a deliberately vulnerable system prompt and a hardened one, to show that model choice moves attack success rates and that prompt-level defenses remain probabilistic.

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

## Prerequisite: Install Ollama and Pull the Models

This demo sends every request to a local [Ollama](https://ollama.com) server. No API key
and no cloud account are involved.

1. **Install Ollama** for macOS, Windows, or Linux by following the official
   instructions: <https://ollama.com/download>. The Ollama quickstart —
   <https://github.com/ollama/ollama/blob/main/README.md#quickstart> — walks through the
   install, the `ollama serve` background service, and running your first model. Model
   pages such as <https://ollama.com/library/llama3.2> list the exact tags and their
   download sizes.

2. **Start the server** (the desktop app does this for you; on Linux run it yourself):

   ```bash
   ollama serve
   ```

3. **Pull the benchmark models before class**, so the notebook does not pause on
   downloads. These are roughly 10 GB in total:

   ```bash
   ollama pull tinyllama:1.1b
   ollama pull smollm2:1.7b
   ollama pull qwen2.5:3b
   ollama pull llama3.2:1b
   ollama pull llama3.2:3b
   ollama pull phi3:mini
   ```

   The notebook skips any model it cannot find, so you can run a subset — but the
   comparison is more interesting with several.

4. **Confirm it is reachable:**

   ```bash
   curl http://127.0.0.1:11434/api/tags
   ```

If Ollama listens somewhere other than `http://127.0.0.1:11434`, set `OLLAMA_API_URL` in
your shell or copy `.env.example` to `.env` and set it there.

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
notebooks/prompt_injection_chat_assistant_local_model.ipynb
```

The notebook runs the same 11 injection payloads against each installed model twice —
once under the deliberately vulnerable task-only system prompt, once under the hardened
prompt — then writes per-model CSVs and a comparison chart to `results/`.
