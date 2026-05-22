# Model Card: EnterpriseSummarizer RAG Assistant

## Model and Pipeline Details

| Field | Value |
|-------|-------|
| System Name | EnterpriseSummarizer |
| Task | Retrieval-augmented document summarization |
| LLM Endpoint | Hosted instruction-following model through internal proxy |
| Embedding Model | `text-embedding-enterprise-small-1` |
| Vector Store | Managed vector database with metadata filters |
| Deployment Status | Limited internal deployment |

## Data Sources

The knowledge base contains HR policy documents, finance approval matrices, engineering incident guides, operations playbooks, and selected executive planning documents. Documents are labeled with department, owner, classification, and allowed groups.

## Known Limitations

- Prompt injection testing has been limited to simple refusal examples.
- Retrieved chunks may contain instructions, templates, comments, or copied email text.
- Document metadata is maintained by document owners and may be incomplete.
- The LLM may summarize sensitive retrieved text even when citations are suppressed.
- Confidence scoring is not available for generated summaries.
- Tool-use authorization is enforced by the application, not by the LLM endpoint.

## Intended Use

The assistant is intended to help employees summarize documents they are already authorized to access. It should not be used to discover restricted documents, bypass department boundaries, produce verbatim confidential excerpts, or export sensitive summaries to unauthorized destinations.
