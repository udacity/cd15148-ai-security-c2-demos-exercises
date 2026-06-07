# Exercise: Build a Prompt Injection Assessment Workflow for a Customer Support Assistant

Estimated time: 40 minutes

## Overview

In this exercise, you will implement a prompt injection assessment workflow against a customer support assistant that uses retrieved documents to answer questions. The assistant represents a simplified RAG system connected to product documentation, internal procedures, and troubleshooting guides.

The goal is to test whether malicious instructions embedded inside indexed documents can manipulate the assistant into revealing restricted information, bypassing policy, or producing unsafe responses.

## Scenario

A software company deploys an AI-powered customer support assistant that processes approximately 15,000 requests daily. The assistant uses a RAG architecture connected to product docs, troubleshooting guides, and internal support procedures.

Security engineers want to determine whether malicious content embedded in indexed documents could alter assistant behavior before the system is rolled out to additional departments.

## Tasks

1. Review the RAG assistant architecture, retrieval flow, and system prompt.
2. Review or create prompt injection payloads for retrieved documents.
3. Insert injection content into indexed documents and rebuild the vector database.
4. Execute support queries that retrieve poisoned content.
5. Evaluate whether the assistant follows injected instructions.
6. Calculate attack success rate across multiple payload styles.
7. Identify weaknesses in retrieval and prompt orchestration.
8. Complete a short assessment report with mitigations.

## Deliverables

Submit:

- A completed notebook implementing the prompt injection assessment workflow.
- At least 3 successful prompt injection examples.
- A comparison table showing successful versus blocked attempts.
- Quantitative attack success measurements.
- A short report summarizing vulnerabilities, operational risks, and mitigations.

## Acceptance Criteria

- The notebook executes injection attempts against the provided RAG system.
- At least 3 injection payloads successfully alter assistant behavior.
- Attack success rate is calculated correctly.
- The final output identifies at least 3 mitigation strategies.
- Results include operational risk and defensive recommendations.

## Expert Tips

- Retrieved documents are untrusted input, even when they come from an internal index.
- Embedded prompt injection is often more realistic than direct user-prompt attacks.
- Separate system instructions from retrieved content whenever possible.
- Output validation and retrieval filtering reduce risk but do not make prompt injection impossible.
