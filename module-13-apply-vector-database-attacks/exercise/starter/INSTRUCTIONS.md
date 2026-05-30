# Exercise: Build a Retrieval Poisoning Assessment Workflow for an Enterprise Search Assistant

Estimated time: 45 minutes

## Scenario

A multinational manufacturing company uses a RAG-based engineering assistant to retrieve maintenance procedures, design specifications, and troubleshooting documentation from a centralized vector database. Security engineers need to assess whether poisoned documents or manipulated embeddings could alter retrieval rankings and produce unsafe operational guidance.

## Tasks

1. Review the RAG architecture in `src/retrieval_poisoning_assessment.py`.
2. Measure clean retrieval behavior for the provided evaluation queries.
3. Implement `create_poisoned_corpus`.
4. Implement `compare_retrieval`.
5. Implement `summarize_assessment`.
6. Rebuild the poisoned vector index.
7. Compare clean versus poisoned ranking behavior with similarity scores.
8. Generate a short assessment report with operational risk and mitigations.

## Acceptance Criteria

- Poisoned content is inserted into the vector database.
- At least 3 retrieval queries demonstrate altered ranking behavior.
- Similarity score comparisons are calculated correctly.
- The report identifies at least 3 mitigation strategies.
- Operational risk is tied to retrieval manipulation, not just generic AI risk.

## Expert Tips

- Retrieval poisoning often succeeds by slightly outperforming legitimate embeddings.
- Smaller chunks can reduce attack blast radius but may increase retrieval noise.
- Monitor provenance and ranking drift to detect suspicious embedding behavior early.
