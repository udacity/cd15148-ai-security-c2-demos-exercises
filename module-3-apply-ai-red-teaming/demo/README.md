# Demo: Build a Red Team Charter Using LLM-Surfaced Attack Vectors

Estimated time: 12 minutes

## Overview

This demo shows a lightweight AI red teaming workflow for a deployed customer-support sentiment analysis service. The workflow uses structured prompts, architecture context, API documentation, model metadata, and sample inference records to ask an LLM endpoint for likely attack vectors.

The goal is not to treat the LLM as an autonomous scanner. The goal is to use it as a structured security analysis assistant that can quickly generate hypotheses for deeper validation.

## Scenario

A security engineering team is reviewing a newly deployed customer-support sentiment analysis service before production rollout. The service processes approximately 50,000 support tickets per day through a public-facing REST API with average response latency of 220 ms. Leadership wants a same-day AI red team assessment to identify likely attack surfaces before the service is connected to downstream automation workflows.

## Demo Flow

1. Load the system documentation from `docs/` and sample inference records from `data/`.
2. Compare a generic vulnerability prompt against the structured prompt in `prompts/structured_vulnerability_discovery_prompt.md`.
3. Review the structural skeletons in `docs/attack_vectors_template.md` and `docs/red_team_charter_template.md` so you know the shape the LLM will fill in.
4. Run the notebook. It writes timestamped artifacts to `outputs/llm_generated_attack_vectors_<YYYYMMDD_HHMMSS>.md` and `outputs/red_team_charter_<YYYYMMDD_HHMMSS>.md`. A reference baseline (`outputs/llm_generated_attack_vectors_baseline.md` and `outputs/red_team_charter_baseline.md`) is committed for comparison.
5. Filter findings into realistic attack paths, lower-confidence assumptions, and items that require validation. Convert the realistic attack vectors into the Objectives and Success Criteria sections of the generated charter.

## Materials

| File | Purpose |
|------|---------|
| `data/sentiment_api_spec.json` | Sample REST API specification for the inference endpoint |
| `data/sample_inference_requests.json` | Five sample inference requests with labeled outputs |
| `docs/deployment_architecture.md` | Text and Mermaid deployment architecture diagram |
| `docs/model_card.md` | Model card with training data, framework, and deployment assumptions |
| `docs/attack_vectors_template.md` | Skeleton template the LLM is asked to follow when generating attack vectors |
| `docs/red_team_charter_template.md` | Skeleton template the LLM is asked to follow when generating the charter |
| `prompts/generic_vulnerability_prompt.md` | Baseline prompt that tends to produce noisy findings |
| `prompts/structured_vulnerability_discovery_prompt.md` | Context-rich prompt for higher-quality findings |
| `outputs/llm_generated_attack_vectors_baseline.md` | Committed reference example of what a "good" attack-vectors run looks like |
| `outputs/red_team_charter_baseline.md` | Committed reference example of what a "good" charter run looks like |
| `notebooks/red_team_charter_demo.ipynb` | Walkthrough notebook for loading materials and assembling the prompt |

## Key Takeaway

LLM-assisted AI red teaming works best when the model is given concrete system context, clear trust boundaries, and threat taxonomy guidance. The output should be treated as hypothesis generation for human review, not as final validation.
