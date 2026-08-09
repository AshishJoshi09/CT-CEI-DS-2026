"""
Entry point for the Streamlit UI.

Run with:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import streamlit as st

from app.services.qa_service import QAService
from app.services.quiz_service import QuizService
from app.services.summary_service import SummaryService
from app.rag.vector_store import VectorStoreError, get_vector_store_manager
from app.ui.chat import render_main
from app.ui.sidebar import render_sidebar
from app.utils.config import has_valid_api_key

st.set_page_config(page_title="AI Study Assistant", page_icon="📚", layout="wide")


def _init_services() -> tuple[QAService, SummaryService, QuizService] | None:
    """Initialize (once per session) the vector store and service layer."""
    if "qa_service" in st.session_state:
        return (
            st.session_state["qa_service"],
            st.session_state["summary_service"],
            st.session_state["quiz_service"],
        )

    try:
        vector_store = get_vector_store_manager()
    except VectorStoreError as exc:
        st.error(f"Failed to start the vector database: {exc}")
        return None

    qa_service = QAService(vector_store)
    summary_service = SummaryService(vector_store)
    quiz_service = QuizService(vector_store)

    st.session_state["qa_service"] = qa_service
    st.session_state["summary_service"] = summary_service
    st.session_state["quiz_service"] = quiz_service
    return qa_service, summary_service, quiz_service


def main() -> None:
    if not has_valid_api_key():
        st.error(
            "⚠️ No OpenAI API key found. Please set `OPENAI_API_KEY` in your `.env` file "
            "(see `.env.example`) and restart the app."
        )
        st.stop()

    services = _init_services()
    if services is None:
        st.stop()
    qa_service, summary_service, quiz_service = services

    render_sidebar(qa_service)
    render_main(qa_service, summary_service, quiz_service)


if __name__ == "__main__":
    main()
