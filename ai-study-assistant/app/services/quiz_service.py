"""
QuizService: generates study/exam-prep artifacts from a document -
MCQs, flashcards, important questions, revision notes, and simple
explanations of a topic.
"""
from __future__ import annotations

import json
import re

from app.models.schemas import FlashcardItem, MCQItem
from app.rag.chain import build_flashcard_chain, build_mcq_chain, get_llm
from app.rag.prompts import (
    IMPORTANT_QUESTIONS_PROMPT,
    REVISION_NOTES_PROMPT,
    SIMPLE_EXPLANATION_PROMPT,
)
from app.rag.vector_store import VectorStoreManager

MAX_CONTENT_CHARS = 40_000


class QuizGenerationError(Exception):
    """Raised when the LLM output cannot be parsed as expected (e.g. malformed JSON)."""


def _extract_json(text: str) -> str:
    """Strip markdown code fences if the model wrapped the JSON in them."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    return match.group(1) if match else text


class QuizService:
    def __init__(self, vector_store: VectorStoreManager) -> None:
        self.vector_store = vector_store
        self.mcq_chain = build_mcq_chain()
        self.flashcard_chain = build_flashcard_chain()
        self.llm = get_llm()

    def _get_document_text(self, doc_id: str) -> str:
        chunks = self.vector_store.get_document_chunk_texts(doc_id)
        if not chunks:
            raise ValueError("No content found for this document. It may have been deleted.")
        return "\n\n".join(chunks)[:MAX_CONTENT_CHARS]

    def generate_mcqs(self, doc_id: str, num_questions: int = 10) -> list[MCQItem]:
        content = self._get_document_text(doc_id)
        raw = self.mcq_chain.invoke({"content": content, "num_questions": num_questions})
        try:
            data = json.loads(_extract_json(raw))
            return [MCQItem(**item) for item in data]
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise QuizGenerationError(
                "The AI produced an invalid quiz format. Please try again."
            ) from exc

    def generate_flashcards(self, doc_id: str, num_cards: int = 10) -> list[FlashcardItem]:
        content = self._get_document_text(doc_id)
        raw = self.flashcard_chain.invoke({"content": content, "num_cards": num_cards})
        try:
            data = json.loads(_extract_json(raw))
            return [FlashcardItem(**item) for item in data]
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise QuizGenerationError(
                "The AI produced an invalid flashcard format. Please try again."
            ) from exc

    def generate_important_questions(self, doc_id: str, num_questions: int = 10) -> str:
        content = self._get_document_text(doc_id)
        chain = IMPORTANT_QUESTIONS_PROMPT | self.llm
        result = chain.invoke({"content": content, "num_questions": num_questions})
        return result.content.strip()

    def generate_revision_notes(self, doc_id: str) -> str:
        content = self._get_document_text(doc_id)
        chain = REVISION_NOTES_PROMPT | self.llm
        result = chain.invoke({"content": content})
        return result.content.strip()

    def explain_simple(self, doc_id: str, topic: str) -> str:
        content = self._get_document_text(doc_id)
        chain = SIMPLE_EXPLANATION_PROMPT | self.llm
        result = chain.invoke({"content": content, "topic": topic})
        return result.content.strip()
