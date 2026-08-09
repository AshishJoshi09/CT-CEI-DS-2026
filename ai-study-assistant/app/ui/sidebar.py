"""
Sidebar: document upload, document management (list/delete/clear),
and configurable RAG settings (chunk size, overlap, top-k, temperature, model).
"""
from __future__ import annotations

import streamlit as st

from app.rag.loader import DocumentLoadError
from app.rag.vector_store import VectorStoreError
from app.services.qa_service import QAService
from app.ui.components import render_document_list
from app.utils.config import settings
from app.utils.helpers import UploadValidationError, save_uploaded_file


def render_sidebar(qa_service: QAService) -> None:
    with st.sidebar:
        st.header("📁 Document Management")

        uploaded_files = st.file_uploader(
            "Upload study material",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            help=f"Max {settings.max_file_size_mb} MB per file. Supported: PDF, DOCX, TXT.",
        )

        if uploaded_files:
            _handle_uploads(qa_service, uploaded_files)

        st.subheader("Uploaded documents")
        render_document_list(qa_service.list_documents(), on_delete=qa_service.delete_document)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧹 Clear all", use_container_width=True):
                qa_service.clear_all_documents()
                st.success("Cleared all documents and chat history.")
                st.rerun()
        with col2:
            if st.button("💬 Reset chat", use_container_width=True):
                qa_service.reset_chat()
                st.rerun()

        st.divider()
        _render_settings()


def _handle_uploads(qa_service: QAService, uploaded_files) -> None:
    """Save + ingest any newly uploaded files, skipping ones already processed this session."""
    processed_names = st.session_state.setdefault("processed_filenames", set())

    for file in uploaded_files:
        if file.name in processed_names:
            continue

        with st.spinner(f"Processing '{file.name}'..."):
            try:
                file_bytes = file.getvalue()
                saved_path = save_uploaded_file(file_bytes, file.name)
                doc_meta = qa_service.ingest_file(saved_path, file.name)
                processed_names.add(file.name)
                st.success(
                    f"'{doc_meta.filename}' indexed: {doc_meta.num_chunks} chunks created."
                )
            except UploadValidationError as exc:
                st.error(f"Upload rejected: {exc}")
            except DocumentLoadError as exc:
                st.error(f"Could not process file: {exc}")
            except VectorStoreError as exc:
                st.error(f"Storage error: {exc}")
            except Exception as exc:  # final safety net for unexpected errors
                st.error(f"Unexpected error while processing '{file.name}': {exc}")


def _render_settings() -> None:
    st.header("⚙️ Settings")

    st.session_state.setdefault("top_k", settings.top_k)
    st.session_state.setdefault("temperature", settings.temperature)

    st.session_state["top_k"] = st.slider(
        "Chunks retrieved per question (top-k)", min_value=1, max_value=10,
        value=st.session_state["top_k"],
    )
    st.session_state["temperature"] = st.slider(
        "LLM temperature", min_value=0.0, max_value=1.0, step=0.1,
        value=st.session_state["temperature"],
    )

    st.caption(f"Model: `{settings.llm_model_name}`")
    st.caption(f"Embeddings: `{settings.embedding_model_name}`")
    st.caption(f"Chunk size / overlap: {settings.chunk_size} / {settings.chunk_overlap}")
