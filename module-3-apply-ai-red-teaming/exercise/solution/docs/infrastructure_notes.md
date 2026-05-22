# Infrastructure Notes

## Runtime

- Python 3.12 FastAPI application
- Hosted LLM accessed through internal proxy
- Managed vector database with metadata filter expressions
- Container image built weekly from pinned dependencies
- Application secrets loaded from enterprise secret manager

## API Gateway Controls

- Bearer token authentication required for all endpoints
- Per-user rate limit: 120 summary requests per hour
- Request body limit: 256 KB
- No separate rate limit for export requests

## Retrieval Configuration

- Top-k retrieval defaults to 8 chunks per request
- Metadata filter uses `department`, `allowed_groups`, and `classification`
- Requests may include up to 20 `document_scope` IDs
- Chunk text and document titles are passed to the LLM prompt

## Logging and Monitoring

- Logs include `user_id`, `department`, `document_scope`, `retrieval_count`, latency, and error messages
- Prompt text is sampled into debug logs at 2 percent during limited deployment
- Export attempts are logged with destination path and status code
- Alerts exist for high error rate and high latency, but not for unusual retrieval patterns
