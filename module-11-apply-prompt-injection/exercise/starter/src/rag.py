"""
RAG pipeline adapted from the project rag_chatbot example.

Flow:
    embed query -> retrieve from vector index -> generate answer from retrieved context

Usage:
    from rag import query_rag
    result = query_rag("What should support do for a refund request?")
"""
from pathlib import Path

from rag_prompt_injection_utils import (
    VulnerableRAGAssistant,
    build_vector_store,
    load_json,
    load_text,
)


ROOT = Path(__file__).resolve().parents[1]
SYSTEM_PROMPT = load_text(ROOT / "prompts" / "rag_system_prompt.md")
_documents = load_json(ROOT / "data" / "support_documents.json")
_index = build_vector_store(_documents)


def retrieve(query, k=2):
    """Retrieve top-k support document chunks from the vector index."""
    return _index.search(query, k=k)


def query_rag(question, k=2):
    """Full RAG pipeline: retrieve context, then generate an answer."""
    assistant = VulnerableRAGAssistant(SYSTEM_PROMPT, _index, top_k=k)
    answer, retrieved = assistant.answer(question)
    return {
        "answer": answer,
        "sources": [doc.id for doc in retrieved],
    }
