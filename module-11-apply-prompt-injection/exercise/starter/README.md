# Starter: RAG Prompt Injection Assessment

Complete the notebook and utility TODOs to assess prompt injection risk in a simplified customer support RAG assistant.

Main files:

- `notebooks/rag_prompt_injection_assessment.ipynb`
- `src/rag.py`
- `src/rag_prompt_injection_utils.py`
- `data/support_documents.json`
- `data/prompt_injection_payload_templates.json`
- `docs/assessment_report_template.md`

The starter uses the same RAG shape as the course project chatbot: embed documents, build a vector index, retrieve top-k chunks, then pass retrieved context to the assistant. FAISS is used when installed; a NumPy fallback keeps the notebook runnable before optional dependencies are installed. The dependency file also includes LangChain, FAISS, sentence-transformers, and OpenAI for extension.
