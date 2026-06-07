# Prompt Injection Assessment Report

## Configuration

- Corpus: sample customer support RAG corpus
- Payload count: 5
- Retrieval top-k: 2
- Assistant mode: deterministic local mock RAG assistant

## Quantitative Results

The solution notebook writes final values to `results/solution_attack_results.csv` and `results/solution_defended_results.csv`.

## Weaknesses Identified

- Retrieved documents were concatenated into the model prompt without clear trust-boundary isolation.
- The vulnerable assistant treated document text as instructions instead of evidence.
- The pipeline lacked retrieval filtering for instruction-like content.
- The pipeline lacked output validation for policy bypass markers.

## Operational Risk

Poisoned support documents could cause the assistant to disclose restricted policy summaries, override escalation requirements, or provide unsafe troubleshooting guidance to customers or employees.

## Mitigations

- Treat retrieved documents as untrusted data and quote them as evidence, not instructions.
- Filter or quarantine documents containing instruction-override patterns.
- Apply output validation for restricted disclosures and unsafe action recommendations.
- Use least-privilege retrieval and separate internal-only corpora from customer-facing corpora.
- Monitor retrieval results and assistant outputs for anomalous policy bypass language.
