"""
QAService: the main entry point used by the UI (and API) for:
  - ingesting uploaded documents into the vector store
  - answering questions via the LangGraph RAG workflow
  - managing document metadata and chat memory for a session
"""
from __future__ import annotations

from pathlib import Path

from app.agents.graph import build_rag_graph
from app.models.schemas import (
    ChatMessage,
    ChatRole,
    DocumentMetadata,
    QAResponse,
    SourceReference,
)
from app.rag.chunker import split_documents
from app.rag.loader import DocumentLoadError, infer_file_type, load_document
from app.rag.vector_store import VectorStoreManager, VectorStoreError
from app.utils.config import settings


class QAService:
    """Stateful service: one instance per user session (held in Streamlit session_state)."""

    def __init__(self, vector_store: VectorStoreManager) -> None:
        self.vector_store = vector_store
        self.graph = build_rag_graph(vector_store)
        self.documents: dict[str, DocumentMetadata] = {}
        self.chat_history: list[ChatMessage] = []

    # --- Ingestion -----------------------------------------------------

    def ingest_file(self, file_path: Path, original_filename: str) -> DocumentMetadata:
        """
        Full ingestion pipeline for one uploaded file:
        load -> chunk -> embed -> store. Returns the resulting metadata record.
        """
        file_type = infer_file_type(original_filename)

        try:
            page_documents = load_document(file_path, file_type)
        except DocumentLoadError:
            raise  # already a user-friendly message

        doc_meta = DocumentMetadata(
            filename=original_filename,
            file_type=file_type,
            num_pages=len(page_documents) if file_type == "pdf" else None,
            size_bytes=file_path.stat().st_size,
        )

        chunks = split_documents(page_documents, doc_id=doc_meta.doc_id, filename=original_filename)
        if not chunks:
            raise DocumentLoadError(f"No usable text could be extracted from '{original_filename}'.")

        try:
            self.vector_store.add_chunks(chunks)
        except VectorStoreError:
            raise

        doc_meta.num_chunks = len(chunks)
        self.documents[doc_meta.doc_id] = doc_meta
        return doc_meta

    # --- Document management --------------------------------------------

    def delete_document(self, doc_id: str) -> None:
        self.vector_store.delete_document(doc_id)
        self.documents.pop(doc_id, None)

    def clear_all_documents(self) -> None:
        self.vector_store.clear_all()
        self.documents.clear()
        self.chat_history.clear()

    def list_documents(self) -> list[DocumentMetadata]:
        return list(self.documents.values())

    # --- Chat memory -------------------------------------------------------

    def _history_as_text(self, max_turns: int = 6) -> str:
        """Render recent chat history as plain text for prompt context."""
        recent = self.chat_history[-max_turns:]
        lines = []
        for msg in recent:
            speaker = "Student" if msg.role == ChatRole.USER else "Assistant"
            lines.append(f"{speaker}: {msg.content}")
        return "\n".join(lines)

    def reset_chat(self) -> None:
        self.chat_history.clear()

    # --- Question answering ------------------------------------------------

    def ask(self, question: str, doc_id: str | None = None, top_k: int | None = None) -> QAResponse:
        """Run the LangGraph RAG workflow for a question and update chat memory."""
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty.")
        if not self.documents:
            return QAResponse(
                answer="Please upload at least one study document before asking a question.",
                sources=[],
                found_in_context=False,
            )

        result = self.graph.invoke(
            {
                "question": question,
                "chat_history": self._history_as_text(),
                "doc_id": doc_id,
                "top_k": top_k or settings.top_k,
            }
        )

        sources = self._build_sources(result.get("retrieved_docs", []), result.get("found_in_context", False))

        self.chat_history.append(ChatMessage(role=ChatRole.USER, content=question))
        self.chat_history.append(
            ChatMessage(role=ChatRole.ASSISTANT, content=result["answer"], sources=sources)
        )

        return QAResponse(
            answer=result["answer"],
            sources=sources,
            found_in_context=result.get("found_in_context", False),
        )

    @staticmethod
    def _build_sources(retrieved_docs, found_in_context: bool) -> list[SourceReference]:
        if not found_in_context:
            return []
        seen = set()
        sources: list[SourceReference] = []
        for doc in retrieved_docs:
            key = (doc.metadata.get("filename"), doc.metadata.get("page"))
            if key in seen:
                continue
            seen.add(key)
            sources.append(
                SourceReference(
                    filename=doc.metadata.get("filename", "unknown"),
                    page=doc.metadata.get("page"),
                    chunk_id=doc.metadata.get("chunk_id", ""),
                    snippet=doc.page_content[:200],
                )
            )
        return sources
