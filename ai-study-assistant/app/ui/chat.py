"""
Main content area: tabbed layout with
  1) Chat - ask questions grounded in uploaded documents
  2) Study Tools - summaries, MCQs, flashcards, revision notes, simple explanations
"""
from __future__ import annotations

import streamlit as st

from app.models.schemas import ChatRole, SummaryRequest, SummaryType
from app.rag.vector_store import VectorStoreError
from app.services.qa_service import QAService
from app.services.quiz_service import QuizGenerationError, QuizService
from app.services.summary_service import SummaryService
from app.ui.components import (
    render_document_picker,
    render_flashcards,
    render_mcqs,
    render_sources,
)


def render_main(qa_service: QAService, summary_service: SummaryService, quiz_service: QuizService) -> None:
    st.title("📚 AI Study Assistant")
    st.caption("Upload your study material and ask anything.")

    tab_chat, tab_tools = st.tabs(["💬 Chat", "🛠️ Study Tools"])

    with tab_chat:
        _render_chat_tab(qa_service)

    with tab_tools:
        _render_study_tools_tab(qa_service, summary_service, quiz_service)


def _render_chat_tab(qa_service: QAService) -> None:
    documents = qa_service.list_documents()
    scope_doc_id = render_document_picker(documents, key="chat_doc_scope") if documents else None

    # Render existing chat history as chat bubbles
    for msg in qa_service.chat_history:
        role = "user" if msg.role == ChatRole.USER else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)
            if msg.role == ChatRole.ASSISTANT:
                render_sources(msg.sources)

    question = st.chat_input("Ask a question about your study material...")
    if question:
        _handle_question(qa_service, question, scope_doc_id)


def _handle_question(qa_service: QAService, question: str, doc_id: str | None) -> None:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                top_k = st.session_state.get("top_k")
                response = qa_service.ask(question, doc_id=doc_id, top_k=top_k)
                st.markdown(response.answer)
                render_sources(response.sources)
            except ValueError as exc:
                st.warning(str(exc))
            except VectorStoreError as exc:
                st.error(f"Retrieval error: {exc}")
            except Exception as exc:
                st.error(f"Something went wrong while answering: {exc}")


def _render_study_tools_tab(
    qa_service: QAService, summary_service: SummaryService, quiz_service: QuizService
) -> None:
    documents = qa_service.list_documents()
    if not documents:
        st.info("Upload a document in the sidebar to use study tools.")
        return

    doc_options = {doc.filename: doc.doc_id for doc in documents}
    selected_name = st.selectbox("Select a document", list(doc_options.keys()))
    doc_id = doc_options[selected_name]

    tool = st.radio(
        "Choose a tool",
        [
            "Summary",
            "MCQ Quiz",
            "Flashcards",
            "Important Questions",
            "Revision Notes",
            "Explain a Topic",
        ],
        horizontal=True,
    )

    st.divider()

    if tool == "Summary":
        _render_summary_tool(summary_service, doc_id)
    elif tool == "MCQ Quiz":
        _render_mcq_tool(quiz_service, doc_id)
    elif tool == "Flashcards":
        _render_flashcard_tool(quiz_service, doc_id)
    elif tool == "Important Questions":
        _render_important_questions_tool(quiz_service, doc_id)
    elif tool == "Revision Notes":
        _render_revision_notes_tool(quiz_service, doc_id)
    elif tool == "Explain a Topic":
        _render_explain_tool(quiz_service, doc_id)


def _render_summary_tool(summary_service: SummaryService, doc_id: str) -> None:
    summary_type = st.selectbox(
        "Summary type",
        options=[t.value for t in SummaryType],
        format_func=lambda v: v.replace("_", " ").title(),
    )
    if st.button("Generate summary", type="primary"):
        with st.spinner("Generating summary..."):
            try:
                result = summary_service.generate(
                    SummaryRequest(doc_id=doc_id, summary_type=SummaryType(summary_type))
                )
                st.markdown(result.content)
            except Exception as exc:
                st.error(f"Could not generate summary: {exc}")


def _render_mcq_tool(quiz_service: QuizService, doc_id: str) -> None:
    num_questions = st.slider("Number of MCQs", 3, 25, 10)
    if st.button("Generate MCQs", type="primary"):
        with st.spinner("Generating MCQs..."):
            try:
                mcqs = quiz_service.generate_mcqs(doc_id, num_questions=num_questions)
                render_mcqs(mcqs)
            except QuizGenerationError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Could not generate MCQs: {exc}")


def _render_flashcard_tool(quiz_service: QuizService, doc_id: str) -> None:
    num_cards = st.slider("Number of flashcards", 3, 25, 10)
    if st.button("Generate flashcards", type="primary"):
        with st.spinner("Generating flashcards..."):
            try:
                cards = quiz_service.generate_flashcards(doc_id, num_cards=num_cards)
                render_flashcards(cards)
            except QuizGenerationError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Could not generate flashcards: {exc}")


def _render_important_questions_tool(quiz_service: QuizService, doc_id: str) -> None:
    num_questions = st.slider("Number of questions", 3, 20, 10)
    if st.button("Generate important questions", type="primary"):
        with st.spinner("Generating questions..."):
            try:
                result = quiz_service.generate_important_questions(doc_id, num_questions=num_questions)
                st.markdown(result)
            except Exception as exc:
                st.error(f"Could not generate questions: {exc}")


def _render_revision_notes_tool(quiz_service: QuizService, doc_id: str) -> None:
    if st.button("Generate revision notes", type="primary"):
        with st.spinner("Generating revision notes..."):
            try:
                result = quiz_service.generate_revision_notes(doc_id)
                st.markdown(result)
            except Exception as exc:
                st.error(f"Could not generate revision notes: {exc}")


def _render_explain_tool(quiz_service: QuizService, doc_id: str) -> None:
    topic = st.text_input("Topic to explain in simple language")
    if st.button("Explain", type="primary") and topic.strip():
        with st.spinner("Explaining..."):
            try:
                result = quiz_service.explain_simple(doc_id, topic.strip())
                st.markdown(result)
            except Exception as exc:
                st.error(f"Could not generate explanation: {exc}")
