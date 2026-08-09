"""
Small reusable Streamlit UI components: source citation rendering,
MCQ rendering, flashcard rendering, and a document picker.
"""
from __future__ import annotations

import streamlit as st

from app.models.schemas import DocumentMetadata, FlashcardItem, MCQItem, SourceReference
from app.utils.helpers import format_bytes


def render_sources(sources: list[SourceReference]) -> None:
    """Render an expandable 'Sources' block under an assistant answer."""
    if not sources:
        return
    with st.expander(f"📎 Sources ({len(sources)})"):
        for src in sources:
            st.markdown(f"**{src.label()}**")
            if src.snippet:
                st.caption(f"\u201c{src.snippet.strip()}...\u201d")
            st.divider()


def render_document_picker(documents: list[DocumentMetadata], key: str = "doc_picker") -> str | None:
    """Render a selectbox to optionally restrict a question/action to one document."""
    if not documents:
        st.info("Upload a document first.")
        return None

    options = {"All documents": None}
    for doc in documents:
        options[f"{doc.filename} ({doc.num_chunks} chunks)"] = doc.doc_id

    label = st.selectbox("Document scope", list(options.keys()), key=key)
    return options[label]


def render_mcqs(mcqs: list[MCQItem]) -> None:
    """Render a list of MCQs with a reveal-answer expander for each."""
    for i, item in enumerate(mcqs, start=1):
        st.markdown(f"**Question {i}: {item.question}**")
        for letter, option_text in item.options.items():
            st.markdown(f"- **{letter}.** {option_text}")
        with st.expander("Show answer & explanation"):
            st.success(f"Correct answer: {item.correct_answer}")
            if item.explanation:
                st.write(item.explanation)
        st.divider()


def render_flashcards(cards: list[FlashcardItem]) -> None:
    """Render flashcards as flip-style expanders."""
    cols = st.columns(2)
    for i, card in enumerate(cards):
        with cols[i % 2]:
            with st.expander(f"🃏 {card.front}"):
                st.write(card.back)


def render_document_list(documents: list[DocumentMetadata], on_delete) -> None:
    """Render the sidebar list of uploaded documents with delete buttons."""
    if not documents:
        st.caption("No documents uploaded yet.")
        return

    for doc in documents:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"**{doc.filename}**")
            page_info = f" · {doc.num_pages} pages" if doc.num_pages else ""
            st.caption(f"{doc.num_chunks} chunks{page_info} · {format_bytes(doc.size_bytes)}")
        with col2:
            if st.button("🗑️", key=f"delete_{doc.doc_id}", help="Delete this document"):
                on_delete(doc.doc_id)
                st.rerun()
