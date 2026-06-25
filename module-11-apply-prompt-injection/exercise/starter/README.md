# Starter: RAG Prompt Injection Assessment

Complete the notebook and report to assess prompt injection risk in a simplified customer support RAG assistant.

Main files:

- `notebooks/rag_prompt_injection_assessment.ipynb`
- `src/rag.py`
- `src/rag_prompt_injection_utils.py`
- `data/support_documents.json`
- `data/prompt_injection_payload_templates.json`
- `docs/assessment_report_template.md`

The starter uses the same RAG shape as the course project chatbot: embed documents with OpenAI `text-embedding-3-small`, build a vector index, retrieve top-k chunks, then pass retrieved context to a live `gpt-4.1-mini` assistant. FAISS is used when installed; a NumPy fallback searches the same OpenAI vectors.

Copy `.env.example` to `.env` or paste the classroom key into the notebook setup cell. Vocareum-issued keys that start with `voc-` automatically use the Vocareum endpoint.
