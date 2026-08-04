# Demo: Build a Malicious Embedding Retrieval Attack Against a RAG System

Estimated time: 13 minutes

## Overview

This demo shows how poisoned documents and manipulated embeddings can alter retrieval rankings in a Retrieval-Augmented Generation (RAG) assistant. Learners compare clean retrieval behavior against an attacked vector index, then observe how untrusted retrieved context can influence a downstream assistant.

The demo runs offline by default with deterministic hashing embeddings and a NumPy vector search fallback. If FAISS is installed, the same code writes a `prebuilt_faiss.index` artifact in `data/`.

## Scenario

A financial services company operates an internal AI research assistant for operational procedures, compliance documentation, and engineering references. During a security review, researchers test whether attacker-controlled documents can be inserted into the retrieval pipeline and surfaced for benign user queries.

## What Students Learn

- Vector databases are security-sensitive infrastructure, not passive storage.
- Poisoned embeddings can rank attacker-controlled content above approved policy.
- Similarity thresholds and top-k choices affect retrieval exposure.
- Retrieved context must be treated as untrusted input.
- Provenance filtering, ingestion validation, and retrieval monitoring reduce risk.

## Run the Demo

```powershell
cd module-13-apply-vector-database-attacks\demo
python run_demo.py
```

The script creates:

- `data/enterprise_documents.json`
- `data/malicious_prompt_injection_documents.json`
- `data/sample_user_queries.json`
- `data/prebuilt_vector_store.npz`
- `data/prebuilt_faiss.index` when FAISS is available
- `results/retrieval_attack_results.json`
- `results/retrieval_attack_results.csv`

To also write a Matplotlib chart, run:

```powershell
python run_demo.py --plot
```

That creates `results/retrieval_ranking_shift.png` when Matplotlib is available and compatible with the installed NumPy version.

## Optional Notebook

Open:

```text
notebooks/malicious_embedding_retrieval_attack_demo.ipynb
```

The notebook uses the same deterministic demo module, so no OpenAI API key or external model download is required for classroom execution.

## What a Real RAG System Needs

The current demo is intentionally a focused security simulation, but a production RAG system would need several additional layers:

- A real embedding model and chunking pipeline for document ingestion.
- A persistent vector store with metadata, provenance, and access controls.
- A retrieval step that returns top-k chunks plus source metadata.
- A prompt-construction step that injects only trusted context into the model prompt.
- Guardrails for provenance checks, source-type filtering, and prompt-injection detection.
- Logging and monitoring for retrieval anomalies, suspicious document sources, and downstream policy violations.

The new helper functions in the demo module show the missing RAG logic in a lightweight form: they build a prompt from retrieved context, detect prompt injection, and reject untrusted context before the model sees it.

## Why Poisoned Vector Databases Matter

A poisoned vector database can surface attacker-controlled documents for benign queries even when the user never intended to retrieve them. In a real RAG system, that makes the model treat untrusted retrieved content as if it were policy-approved context. The result can be credential disclosure, bypass guidance, or instruction override. The demo illustrates that risk by inserting manipulated documents whose embeddings are blended toward the target query vector.

## Optional Live Integrations

The requirements file includes OpenAI, LangChain, FAISS, and sentence-transformers for instructors who want to extend the lab with live embeddings or a full RAG chain. The shipped demo keeps those dependencies optional so the core attack workflow remains reproducible in restricted environments.
