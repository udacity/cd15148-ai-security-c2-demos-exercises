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

## Optional Live Integrations

The requirements file includes OpenAI, LangChain, FAISS, and sentence-transformers for instructors who want to extend the lab with live embeddings or a full RAG chain. The shipped demo keeps those dependencies optional so the core attack workflow remains reproducible in restricted environments.
