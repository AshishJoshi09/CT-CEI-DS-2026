"""
Pydantic data models shared across the RAG pipeline, services, and UI layer.
Keeping these in one place avoids circular imports and keeps request/response
shapes consistent.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


def new_id() -> str:
    """Generate a short unique identifier."""
    return uuid.uuid4().hex[:12]


class DocumentMetadata(BaseModel):
    """Metadata stored for every uploaded document."""

    doc_id: str = Field(default_factory=new_id)
    filename: str
    file_type: str  # "pdf" | "docx" | "txt"
    num_pages: Optional[int] = None
    num_chunks: int = 0
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    size_bytes: int = 0


class ChunkMetadata(BaseModel):
    """Metadata attached to each chunk stored in the vector database."""

    doc_id: str
    chunk_id: str
    filename: str
    page: Optional[int] = None  # 1-indexed page number if available


class SourceReference(BaseModel):
    """A single citation shown alongside an answer."""

    filename: str
    page: Optional[int] = None
    chunk_id: str
    snippet: str = ""

    def label(self) -> str:
        """Human-readable citation label, e.g. 'notes.pdf, Page 4'."""
        if self.page is not None:
            return f"{self.filename}, Page {self.page}"
        return self.filename


class ChatRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """A single turn in the chat history."""

    role: ChatRole
    content: str
    sources: list[SourceReference] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class QARequest(BaseModel):
    question: str
    doc_id: Optional[str] = None  # restrict retrieval to one document if set
    top_k: Optional[int] = None


class QAResponse(BaseModel):
    answer: str
    sources: list[SourceReference] = Field(default_factory=list)
    found_in_context: bool = True


class SummaryType(str, Enum):
    SHORT = "short"
    DETAILED = "detailed"
    KEY_POINTS = "key_points"
    DEFINITIONS = "definitions"
    FORMULAS = "formulas"


class SummaryRequest(BaseModel):
    doc_id: str
    summary_type: SummaryType = SummaryType.SHORT


class SummaryResponse(BaseModel):
    doc_id: str
    summary_type: SummaryType
    content: str


class MCQItem(BaseModel):
    question: str
    options: dict[str, str]  # {"A": "...", "B": "...", ...}
    correct_answer: str  # e.g. "B"
    explanation: str = ""


class FlashcardItem(BaseModel):
    front: str
    back: str


class QuizRequest(BaseModel):
    doc_id: str
    num_questions: int = Field(default=10, gt=0, le=50)


class FlashcardRequest(BaseModel):
    doc_id: str
    num_cards: int = Field(default=10, gt=0, le=50)
