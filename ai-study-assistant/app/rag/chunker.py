"""
Splits loaded Document objects into smaller overlapping chunks suitable
for embedding, and attaches consistent metadata (doc_id, chunk_id, filename, page).
"""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.models.schemas import new_id
from app.utils.config import settings


def split_documents(
    documents: list[Document],
    doc_id: str,
    filename: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """
    Split a list of page-level Documents into chunk-level Documents.

    Each output chunk carries metadata:
        - doc_id: the parent document's id
        - chunk_id: a unique id for this chunk
        - filename: original filename (for citations)
        - page: page number if known (PDFs), else None
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Document] = []
    for page_doc in documents:
        page_number = page_doc.metadata.get("page")
        split_texts = splitter.split_text(page_doc.page_content)
        for text in split_texts:
            if not text.strip():
                continue
            chunk = Document(
                page_content=text,
                metadata={
                    "doc_id": doc_id,
                    "chunk_id": new_id(),
                    "filename": filename,
                    "page": page_number,
                },
            )
            chunks.append(chunk)

    return chunks
