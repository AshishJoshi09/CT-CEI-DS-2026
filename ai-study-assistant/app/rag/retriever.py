"""
Retrieval helpers built on top of VectorStoreManager. Kept separate from
vector_store.py so retrieval-specific logic (top-k, filtering, formatting)
doesn't bloat the storage wrapper.
"""
from __future__ import annotations

from langchain_core.documents import Document

from app.rag.vector_store import VectorStoreManager
from app.utils.config import settings


def retrieve_chunks(
    vector_store: VectorStoreManager,
    query: str,
    top_k: int | None = None,
    doc_id: str | None = None,
) -> list[Document]:
    """
    Retrieve the most relevant chunks for a query.

    Args:
        vector_store: the shared VectorStoreManager.
        query: the user's natural-language question.
        top_k: number of chunks to retrieve (defaults to settings.top_k).
        doc_id: if set, restrict retrieval to a single document.
    """
    k = top_k or settings.top_k
    return vector_store.similarity_search(query=query, top_k=k, doc_id=doc_id)


def format_context(chunks: list[Document]) -> str:
    """
    Build a single context string from retrieved chunks, tagging each chunk
    with its source so the LLM can cite it accurately.
    """
    if not chunks:
        return ""

    parts = []
    for i, chunk in enumerate(chunks, start=1):
        filename = chunk.metadata.get("filename", "unknown")
        page = chunk.metadata.get("page")
        source_tag = f"{filename}" + (f", Page {page}" if page else "")
        parts.append(f"[Source {i}: {source_tag}]\n{chunk.page_content}")

    return "\n\n".join(parts)
