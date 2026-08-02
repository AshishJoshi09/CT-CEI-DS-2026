"""
Stage 2: Text Chunking
Splits raw document text into smaller, overlapping chunks so retrieval can pinpoint
relevant sections instead of returning whole pages/documents.
"""

from dataclasses import dataclass
from document_loader import RawDocument


@dataclass
class Chunk:
    text: str
    source: str
    page: int
    chunk_id: int


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Simple sliding-window character splitter with sentence-boundary snapping,
    so chunks don't get cut mid-sentence when avoidable.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        segment = text[start:end]

        # try to end on a sentence boundary if one exists near the end
        if end < len(text):
            last_period = segment.rfind(". ")
            if last_period != -1 and last_period > chunk_size * 0.5:
                end = start + last_period + 1
                segment = text[start:end]

        chunks.append(segment.strip())
        start = end - chunk_overlap  # step forward, keeping overlap

    return [c for c in chunks if c]


def chunk_documents(
    raw_docs: list[RawDocument], chunk_size: int, chunk_overlap: int
) -> list[Chunk]:
    chunks = []
    chunk_id = 0
    for doc in raw_docs:
        for piece in split_text(doc.text, chunk_size, chunk_overlap):
            chunks.append(
                Chunk(text=piece, source=doc.source, page=doc.page, chunk_id=chunk_id)
            )
            chunk_id += 1

    print(f"Created {len(chunks)} chunk(s).")
    return chunks
