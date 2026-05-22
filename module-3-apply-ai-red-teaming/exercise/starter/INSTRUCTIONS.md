# Exercise: Build an AI Red Team Workflow for a RAG Summarization Service

Estimated time: 35 minutes

## Overview

In this exercise, you will build an AI red team assessment workflow for a generative AI document summarization service deployed inside an enterprise environment. You will use provided system documentation, sample deployment artifacts, and a notebook scaffold to generate two artifacts:

1. A proposed attack vector list
2. A red team charter based on those attack vectors

This exercise extends the demo by adding constraints around risk prioritization, false-positive reduction, and mitigation planning.

## Scenario

An enterprise operations team recently deployed an internal document summarization assistant that processes approximately 12,000 documents per day across HR, finance, and engineering departments. The service uses a Retrieval-Augmented Generation pipeline connected to sensitive internal documents and has been approved for limited deployment.

A security review identified concerns about prompt injection, data leakage, insecure tool usage, and unauthorized retrieval access. Your task is to perform an AI red team assessment and generate a prioritized vulnerability report before the system is expanded company-wide.

## Your Tasks

1. Review the fictional RAG system materials in `docs/` and `data/`.
2. Complete `prompts/structured_vulnerability_discovery_prompt.md`.
3. Open `notebooks/rag_red_team_workflow_exercise.ipynb`.
4. Run the first LLM request to generate proposed attacks and save `docs/llm_generated_attack_vectors.md`.
5. Run the second LLM request to append the generated attacks to the context bundle and save `docs/red_team_charter.md`.
6. Review both artifacts for false positives, unsupported assumptions, missing mitigations, and operational risk.

## Starter Materials

| File | Purpose |
|------|---------|
| `data/rag_api_spec.json` | Sample REST API specification for the summarization service |
| `data/sample_summary_requests.json` | Five sample summarization requests and outputs |
| `docs/deployment_architecture.md` | RAG deployment diagram and trust boundaries |
| `docs/model_card.md` | Model, retrieval, and data assumptions |
| `docs/infrastructure_notes.md` | Fictional deployment and infrastructure configuration notes |
| `docs/retrieval_access_policy.md` | Intended document-access rules |
| `prompts/structured_vulnerability_discovery_prompt.md` | Prompt scaffold you must complete |
| `notebooks/rag_red_team_workflow_exercise.ipynb` | Notebook scaffold for the two-request workflow |

## Required Outputs

By the end of the exercise, your `starter/docs/` folder should contain:

- `llm_generated_attack_vectors.md`
- `red_team_charter.md`

## Key Takeaway

AI red teaming is not just about generating attacks. It is about producing actionable, prioritized security findings that align technical vulnerabilities with operational risk.
