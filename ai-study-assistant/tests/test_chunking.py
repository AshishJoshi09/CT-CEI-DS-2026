"""
Tests for app.rag.chunker - verifies chunk metadata and splitting behavior.
"""
from __future__ import annotations

from langchain_core.documents import Document

from app.rag.chunker import split_documents


def test_split_documents_basic():
    long_text = "This is a sentence about supervised learning. " * 100
    page_docs = [Document(page_content=long_text, metadata={"page": 1})]

    chunks = split_documents(
        page_docs, doc_id="doc123", filename="notes.pdf", chunk_size=200, chunk_overlap=50
    )

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.metadata["doc_id"] == "doc123"
        assert chunk.metadata["filename"] == "notes.pdf"
        assert chunk.metadata["page"] == 1
        assert "chunk_id" in chunk.metadata
        assert len(chunk.page_content) > 0


def test_split_documents_unique_chunk_ids():
    page_docs = [
        Document(page_content="Paragraph one. " * 50, metadata={"page": 1}),
        Document(page_content="Paragraph two. " * 50, metadata={"page": 2}),
    ]

    chunks = split_documents(page_docs, doc_id="doc456", filename="book.pdf")
    chunk_ids = [c.metadata["chunk_id"] for c in chunks]

    assert len(chunk_ids) == len(set(chunk_ids))  # all unique


def test_split_documents_preserves_page_numbers():
    page_docs = [
        Document(page_content="Content on page five. " * 30, metadata={"page": 5}),
    ]

    chunks = split_documents(page_docs, doc_id="doc789", filename="report.pdf")

    assert all(c.metadata["page"] == 5 for c in chunks)


def test_split_documents_empty_input():
    chunks = split_documents([], doc_id="doc000", filename="empty.txt")
    assert chunks == []
