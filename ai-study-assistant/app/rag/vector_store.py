"""
Thin wrapper around a persistent Chroma vector store.

Exposes only the operations the app needs: add chunks, similarity search
(optionally filtered to one document), list distinct documents, delete a
document's chunks, and clear the entire collection.
"""
from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.rag.embeddings import get_embeddings
from app.utils.config import settings


class VectorStoreError(Exception):
    """Raised when a vector store operation fails."""


class VectorStoreManager:
    """Manages a single persistent Chroma collection for the study assistant."""

    def __init__(self) -> None:
        try:
            self._store = Chroma(
                collection_name=settings.chroma_collection_name,
                embedding_function=get_embeddings(),
                persist_directory=str(settings.chroma_dir),
            )
        except Exception as exc:
            raise VectorStoreError(f"Failed to initialize the vector database: {exc}") from exc

    @property
    def store(self) -> Chroma:
        return self._store

    def add_chunks(self, chunks: list[Document]) -> None:
        """Embed and persist a list of chunk Documents."""
        if not chunks:
            return
        try:
            ids = [c.metadata["chunk_id"] for c in chunks]
            self._store.add_documents(documents=chunks, ids=ids)
        except Exception as exc:
            raise VectorStoreError(f"Failed to store document embeddings: {exc}") from exc

    def similarity_search(
        self, query: str, top_k: int, doc_id: str | None = None
    ) -> list[Document]:
        """Return the top_k most relevant chunks, optionally restricted to one doc_id."""
        try:
            search_filter = {"doc_id": doc_id} if doc_id else None
            return self._store.similarity_search(query, k=top_k, filter=search_filter)
        except Exception as exc:
            raise VectorStoreError(f"Retrieval from the vector database failed: {exc}") from exc

    def delete_document(self, doc_id: str) -> None:
        """Delete all chunks belonging to a given document id."""
        try:
            self._store.delete(where={"doc_id": doc_id})
        except Exception as exc:
            raise VectorStoreError(f"Failed to delete document '{doc_id}': {exc}") from exc

    def clear_all(self) -> None:
        """Delete the entire collection's contents."""
        try:
            existing = self._store.get()
            ids = existing.get("ids", [])
            if ids:
                self._store.delete(ids=ids)
        except Exception as exc:
            raise VectorStoreError(f"Failed to clear the vector database: {exc}") from exc

    def get_document_chunk_texts(self, doc_id: str, limit: int = 400) -> list[str]:
        """Fetch all chunk texts for a document (used for summarization/quiz generation)."""
        try:
            result = self._store.get(where={"doc_id": doc_id}, limit=limit)
            return result.get("documents", []) or []
        except Exception as exc:
            raise VectorStoreError(f"Failed to fetch document content: {exc}") from exc


# Module-level singleton so the whole app shares one Chroma connection.
_manager: VectorStoreManager | None = None


def get_vector_store_manager() -> VectorStoreManager:
    """Return a process-wide singleton VectorStoreManager."""
    global _manager
    if _manager is None:
        _manager = VectorStoreManager()
    return _manager
