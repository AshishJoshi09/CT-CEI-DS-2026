"""
SummaryService: generates study aids (summaries, key points, definitions,
formulas) for a single document, using its full stored content rather than
a similarity search - summarization needs the whole document, not just the
top-k most relevant chunks.
"""
from __future__ import annotations

from app.models.schemas import SummaryRequest, SummaryResponse, SummaryType
from app.rag.chain import build_summary_chain
from app.rag.prompts import SUMMARY_PROMPTS
from app.rag.vector_store import VectorStoreManager

# Rough safety cap on characters sent to the LLM in one call, to avoid
# blowing past context limits on very large documents.
MAX_CONTENT_CHARS = 40_000


class SummaryService:
    def __init__(self, vector_store: VectorStoreManager) -> None:
        self.vector_store = vector_store
        self.chain = build_summary_chain()

    def _get_document_text(self, doc_id: str) -> str:
        chunks = self.vector_store.get_document_chunk_texts(doc_id)
        if not chunks:
            raise ValueError("No content found for this document. It may have been deleted.")
        text = "\n\n".join(chunks)
        return text[:MAX_CONTENT_CHARS]

    def generate(self, request: SummaryRequest) -> SummaryResponse:
        content = self._get_document_text(request.doc_id)
        instruction = SUMMARY_PROMPTS[request.summary_type.value]

        result = self.chain.invoke({"instruction": instruction, "content": content}).strip()

        return SummaryResponse(
            doc_id=request.doc_id,
            summary_type=request.summary_type,
            content=result,
        )

    def generate_all(self, doc_id: str) -> dict[SummaryType, str]:
        """Convenience method: generate every summary type at once."""
        content = self._get_document_text(doc_id)
        results: dict[SummaryType, str] = {}
        for summary_type, instruction in SUMMARY_PROMPTS.items():
            results[SummaryType(summary_type)] = self.chain.invoke(
                {"instruction": instruction, "content": content}
            ).strip()
        return results
