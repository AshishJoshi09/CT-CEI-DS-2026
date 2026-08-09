"""
Tests for retrieval, formatting, and the end-to-end RAG pipeline.

Unit tests use a mocked vector store so they run without any API key or
network access. Integration tests that exercise the real OpenAI API and
ChromaDB are marked and automatically skipped unless a valid
OPENAI_API_KEY is present in the environment.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from app.rag.retriever import format_context, retrieve_chunks
from app.utils.config import has_valid_api_key

requires_api_key = pytest.mark.skipif(
    not has_valid_api_key(), reason="OPENAI_API_KEY not configured - skipping integration test"
)


# --- Unit tests (no API key required) ---------------------------------------


def test_format_context_empty():
    assert format_context([]) == ""


def test_format_context_includes_source_tags():
    chunks = [
        Document(page_content="Deadlock is a state where processes wait forever.",
                 metadata={"filename": "os.pdf", "page": 3}),
        Document(page_content="Deadlock can be prevented using resource ordering.",
                 metadata={"filename": "os.pdf", "page": 4}),
    ]

    context = format_context(chunks)

    assert "os.pdf, Page 3" in context
    assert "os.pdf, Page 4" in context
    assert "Deadlock is a state" in context


def test_retrieve_chunks_uses_top_k_and_doc_id():
    mock_store = MagicMock()
    mock_store.similarity_search.return_value = [
        Document(page_content="chunk text", metadata={"filename": "ml.pdf", "page": 1})
    ]

    results = retrieve_chunks(mock_store, query="What is overfitting?", top_k=3, doc_id="doc-abc")

    mock_store.similarity_search.assert_called_once_with(
        query="What is overfitting?", top_k=3, doc_id="doc-abc"
    )
    assert len(results) == 1


def test_build_sources_deduplicates_by_filename_and_page():
    from app.services.qa_service import QAService

    docs = [
        Document(page_content="a", metadata={"filename": "dbms.pdf", "page": 2, "chunk_id": "c1"}),
        Document(page_content="b", metadata={"filename": "dbms.pdf", "page": 2, "chunk_id": "c2"}),
        Document(page_content="c", metadata={"filename": "dbms.pdf", "page": 3, "chunk_id": "c3"}),
    ]

    sources = QAService._build_sources(docs, found_in_context=True)

    assert len(sources) == 2  # page 2 duplicate collapsed
    assert sources[0].label() == "dbms.pdf, Page 2"


def test_build_sources_empty_when_not_found_in_context():
    from app.services.qa_service import QAService

    docs = [Document(page_content="a", metadata={"filename": "x.pdf", "page": 1, "chunk_id": "c1"})]
    sources = QAService._build_sources(docs, found_in_context=False)

    assert sources == []


# --- Integration tests (require OPENAI_API_KEY) -----------------------------


@requires_api_key
def test_end_to_end_ingestion_and_qa(tmp_path):
    """
    Full pipeline smoke test: ingest a TXT file, ask a question, verify the
    answer is grounded and cites the correct source file.
    Requires a real OPENAI_API_KEY and network access.
    """
    from app.rag.vector_store import VectorStoreManager
    from app.services.qa_service import QAService
    from app.utils.config import settings

    # Use an isolated Chroma collection/dir for this test run
    settings.chroma_collection_name = "test_collection_qa"

    file_path = tmp_path / "ml_notes.txt"
    file_path.write_text(
        "Supervised learning is a machine learning approach where a model "
        "learns from labeled training data to make predictions."
    )

    vector_store = VectorStoreManager()
    service = QAService(vector_store)
    service.ingest_file(file_path, "ml_notes.txt")

    response = service.ask("What is supervised learning?")

    assert response.found_in_context is True
    assert len(response.sources) > 0
    assert response.sources[0].filename == "ml_notes.txt"

    vector_store.clear_all()


@requires_api_key
def test_not_found_when_question_unrelated(tmp_path):
    from app.rag.vector_store import VectorStoreManager
    from app.services.qa_service import QAService
    from app.utils.config import settings

    settings.chroma_collection_name = "test_collection_notfound"

    file_path = tmp_path / "topic.txt"
    file_path.write_text("Photosynthesis converts sunlight into chemical energy in plants.")

    vector_store = VectorStoreManager()
    service = QAService(vector_store)
    service.ingest_file(file_path, "topic.txt")

    response = service.ask("What is the capital of France?")

    assert response.found_in_context is False
    assert "not find" in response.answer.lower() or "could not" in response.answer.lower()

    vector_store.clear_all()


@requires_api_key
def test_document_specific_retrieval(tmp_path):
    """Verify that restricting to one doc_id excludes chunks from other documents."""
    from app.rag.vector_store import VectorStoreManager
    from app.services.qa_service import QAService
    from app.utils.config import settings

    settings.chroma_collection_name = "test_collection_multidoc"

    file_a = tmp_path / "dbms.txt"
    file_a.write_text("A deadlock in DBMS occurs when transactions wait on each other's locks.")
    file_b = tmp_path / "os.txt"
    file_b.write_text("An operating system deadlock occurs when processes hold and wait for resources.")

    vector_store = VectorStoreManager()
    service = QAService(vector_store)
    meta_a = service.ingest_file(file_a, "dbms.txt")
    service.ingest_file(file_b, "os.txt")

    response = service.ask("Explain deadlock.", doc_id=meta_a.doc_id)

    assert response.found_in_context is True
    assert all(src.filename == "dbms.txt" for src in response.sources)

    vector_store.clear_all()
