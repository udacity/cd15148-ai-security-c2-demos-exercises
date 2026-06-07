# Solution: RAG Prompt Injection Assessment

This solution implements a prompt injection assessment workflow against a simplified customer support RAG assistant.

The completed workflow:

- Loads a sample support document corpus.
- Builds a FAISS-backed retrieval index using deterministic local embeddings.
- Provides a project-style `src/rag.py` with `retrieve()` and `query_rag()`.
- Inserts malicious instructions into selected indexed documents.
- Runs benign support queries that retrieve poisoned content.
- Measures unsafe assistant compromise rate.
- Compares against a guarded assistant that filters retrieved injection attempts.
- Writes reference CSV results to `results/`.

The reference solution is deterministic and does not require an API key.

Reference results:

| Assistant | Attempts | Successful Injections | Attack Success Rate |
| --- | ---: | ---: | ---: |
| Vulnerable RAG assistant | 5 | 5 | 1.00 |
| Guarded RAG assistant | 5 | 0 | 0.00 |
