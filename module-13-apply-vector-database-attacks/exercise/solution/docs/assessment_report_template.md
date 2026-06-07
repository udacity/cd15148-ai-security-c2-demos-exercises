# Retrieval Poisoning Assessment Report

## Executive Summary

The completed workflow measures clean and poisoned retrieval behavior, then reports altered rankings, downstream compromised responses, and operational risk.

## Recommended Mitigations

- Require provenance validation and signed ingestion records for indexed documents.
- Monitor retrieval ranking drift, source mix, and sudden similarity-score changes.
- Filter or quarantine unverified documents before they enter production indexes.
- Separate retrieved context from system instructions and reject instruction-like retrieved text.
- Use human review for high-risk maintenance domains before expanding assistant coverage.
