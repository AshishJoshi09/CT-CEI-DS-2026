"""
LangGraph workflow for grounded question answering.

Why LangGraph here (and not just a linear LCEL chain): the pipeline needs a
conditional branch - if retrieved context is insufficient, we must short-circuit
straight to a "not found" answer instead of letting the LLM generate from thin
air. That branching + explicit state is exactly what LangGraph is good for.

Workflow:

    START
      |
    understand_query      (rewrite follow-up question into a standalone question)
      |
    retrieve_documents     (vector similarity search)
      |
    check_context           (is the retrieved context enough to answer?)
      |
      +--(insufficient)--> no_context_answer --> END
      |
      +--(sufficient)----> generate_answer --> validate_answer --> END
"""
from __future__ import annotations

from typing import Optional, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, StateGraph

from app.rag.chain import (
    build_condense_question_chain,
    build_context_check_chain,
    build_qa_chain,
)
from app.rag.prompts import NOT_FOUND_PHRASE
from app.rag.retriever import format_context, retrieve_chunks
from app.rag.vector_store import VectorStoreManager
from app.utils.config import settings


class RAGState(TypedDict, total=False):
    """Shared state passed between LangGraph nodes."""

    question: str
    standalone_question: str
    chat_history: str
    doc_id: Optional[str]
    top_k: int
    retrieved_docs: list[Document]
    context_text: str
    context_sufficient: bool
    answer: str
    found_in_context: bool


def _node_understand_query(state: RAGState) -> RAGState:
    """Rewrite the question to be standalone, resolving pronouns via chat history."""
    if not state.get("chat_history", "").strip():
        # No history yet -> nothing to resolve, skip the extra LLM call.
        return {**state, "standalone_question": state["question"]}

    chain = build_condense_question_chain()
    rewritten = chain.invoke(
        {"chat_history": state["chat_history"], "question": state["question"]}
    ).strip()
    return {**state, "standalone_question": rewritten or state["question"]}


def _make_retrieve_node(vector_store: VectorStoreManager):
    def _node_retrieve_documents(state: RAGState) -> RAGState:
        query = state.get("standalone_question") or state["question"]
        docs = retrieve_chunks(
            vector_store=vector_store,
            query=query,
            top_k=state.get("top_k") or settings.top_k,
            doc_id=state.get("doc_id"),
        )
        return {**state, "retrieved_docs": docs, "context_text": format_context(docs)}

    return _node_retrieve_documents


def _node_check_context(state: RAGState) -> RAGState:
    """Decide if the retrieved context is sufficient to answer the question."""
    if not state.get("retrieved_docs"):
        return {**state, "context_sufficient": False}

    chain = build_context_check_chain()
    verdict = chain.invoke(
        {
            "context": state["context_text"],
            "question": state.get("standalone_question") or state["question"],
        }
    ).strip().upper()
    return {**state, "context_sufficient": verdict.startswith("Y")}


def _route_after_check(state: RAGState) -> str:
    return "generate_answer" if state.get("context_sufficient") else "no_context_answer"


def _node_generate_answer(state: RAGState) -> RAGState:
    chain = build_qa_chain()
    answer = chain.invoke(
        {
            "context": state["context_text"],
            "chat_history": state.get("chat_history", ""),
            "question": state.get("standalone_question") or state["question"],
        }
    ).strip()
    return {**state, "answer": answer, "found_in_context": True}


def _node_no_context_answer(state: RAGState) -> RAGState:
    return {**state, "answer": NOT_FOUND_PHRASE, "found_in_context": False}


def _node_validate_answer(state: RAGState) -> RAGState:
    """
    Final guardrail: if the model produced an empty answer, or itself claims
    it doesn't know, normalize to the standard "not found" message so the UI
    behaves consistently.
    """
    answer = state.get("answer", "").strip()
    if not answer:
        return {**state, "answer": NOT_FOUND_PHRASE, "found_in_context": False}
    return state


def build_rag_graph(vector_store: VectorStoreManager):
    """Compile and return the LangGraph RAG workflow."""
    graph = StateGraph(RAGState)

    graph.add_node("understand_query", _node_understand_query)
    graph.add_node("retrieve_documents", _make_retrieve_node(vector_store))
    graph.add_node("check_context", _node_check_context)
    graph.add_node("generate_answer", _node_generate_answer)
    graph.add_node("no_context_answer", _node_no_context_answer)
    graph.add_node("validate_answer", _node_validate_answer)

    graph.set_entry_point("understand_query")
    graph.add_edge("understand_query", "retrieve_documents")
    graph.add_edge("retrieve_documents", "check_context")
    graph.add_conditional_edges(
        "check_context",
        _route_after_check,
        {"generate_answer": "generate_answer", "no_context_answer": "no_context_answer"},
    )
    graph.add_edge("generate_answer", "validate_answer")
    graph.add_edge("no_context_answer", END)
    graph.add_edge("validate_answer", END)

    return graph.compile()
