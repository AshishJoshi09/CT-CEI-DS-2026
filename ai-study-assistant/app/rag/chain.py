"""
Builds the LLM client and the LCEL (LangChain Expression Language) chains
used for question answering, summarization, quiz, and flashcard generation.

Kept separate from the LangGraph workflow (agents/graph.py) - the graph
orchestrates *when* these chains run, this module defines *what* they do.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from app.rag.prompts import (
    CONDENSE_QUESTION_PROMPT,
    CONTEXT_CHECK_PROMPT,
    FLASHCARD_PROMPT,
    MCQ_PROMPT,
    QA_PROMPT,
    SUMMARY_PROMPT,
)
from app.utils.config import settings


@lru_cache(maxsize=1)
def get_llm(temperature: float | None = None) -> ChatOpenAI:
    """Return a cached ChatOpenAI client. Cached per default temperature."""
    return ChatOpenAI(
        model=settings.llm_model_name,
        temperature=temperature if temperature is not None else settings.temperature,
        api_key=settings.openai_api_key,
    )


def build_qa_chain() -> Runnable:
    """LCEL chain: prompt -> LLM -> string output, for grounded question answering."""
    return QA_PROMPT | get_llm() | StrOutputParser()


def build_context_check_chain() -> Runnable:
    """LCEL chain that judges whether retrieved context is sufficient (YES/NO)."""
    # Low temperature for a more deterministic judgment call.
    return CONTEXT_CHECK_PROMPT | get_llm(temperature=0.0) | StrOutputParser()


def build_condense_question_chain() -> Runnable:
    """LCEL chain that rewrites a follow-up question into a standalone question."""
    return CONDENSE_QUESTION_PROMPT | get_llm(temperature=0.0) | StrOutputParser()


def build_summary_chain() -> Runnable:
    """LCEL chain for all summary/study-aid text generation (non-JSON outputs)."""
    return SUMMARY_PROMPT | get_llm() | StrOutputParser()


def build_mcq_chain() -> Runnable:
    """LCEL chain that generates MCQs as a JSON string."""
    return MCQ_PROMPT | get_llm() | StrOutputParser()


def build_flashcard_chain() -> Runnable:
    """LCEL chain that generates flashcards as a JSON string."""
    return FLASHCARD_PROMPT | get_llm() | StrOutputParser()
