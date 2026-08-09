"""
Optional FastAPI backend exposing the same RAG pipeline as REST endpoints.

The primary UI is the Streamlit app (streamlit_app.py), which calls the
service layer directly. This API is provided so the same backend can be
used programmatically or from a different frontend, and to demonstrate
the FastAPI layer requested in the project spec.

Run with:
    uvicorn app.main:app --reload
"""
from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.models.schemas import (
    DocumentMetadata,
    FlashcardRequest,
    QARequest,
    QAResponse,
    QuizRequest,
    SummaryRequest,
    SummaryResponse,
)
from app.rag.loader import DocumentLoadError
from app.rag.vector_store import VectorStoreError, get_vector_store_manager
from app.services.qa_service import QAService
from app.services.quiz_service import QuizGenerationError, QuizService
from app.services.summary_service import SummaryService
from app.utils.config import has_valid_api_key
from app.utils.helpers import UploadValidationError, save_uploaded_file

app = FastAPI(title="AI Study Assistant API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared backend instance for the API (simple demo-scale setup;
# a production deployment would scope these per authenticated user/session).
_vector_store = get_vector_store_manager()
qa_service = QAService(_vector_store)
summary_service = SummaryService(_vector_store)
quiz_service = QuizService(_vector_store)


@app.on_event("startup")
def _check_config() -> None:
    if not has_valid_api_key():
        # Don't crash the server - just warn; endpoints will fail clearly per-request.
        print("WARNING: OPENAI_API_KEY is not set. Requests to the LLM will fail.")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "api_key_configured": has_valid_api_key()}


@app.post("/documents/upload", response_model=DocumentMetadata)
async def upload_document(file: UploadFile = File(...)) -> DocumentMetadata:
    try:
        file_bytes = await file.read()
        saved_path = save_uploaded_file(file_bytes, file.filename)
        return qa_service.ingest_file(saved_path, file.filename)
    except (UploadValidationError, DocumentLoadError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except VectorStoreError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/documents", response_model=list[DocumentMetadata])
def list_documents() -> list[DocumentMetadata]:
    return qa_service.list_documents()


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str) -> dict:
    qa_service.delete_document(doc_id)
    return {"status": "deleted", "doc_id": doc_id}


@app.delete("/documents")
def clear_documents() -> dict:
    qa_service.clear_all_documents()
    return {"status": "cleared"}


@app.post("/ask", response_model=QAResponse)
def ask_question(request: QARequest) -> QAResponse:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        return qa_service.ask(request.question, doc_id=request.doc_id, top_k=request.top_k)
    except VectorStoreError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/summary", response_model=SummaryResponse)
def generate_summary(request: SummaryRequest) -> SummaryResponse:
    try:
        return summary_service.generate(request)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/quiz/mcq")
def generate_mcqs(request: QuizRequest) -> dict:
    try:
        mcqs = quiz_service.generate_mcqs(request.doc_id, num_questions=request.num_questions)
        return {"questions": [m.model_dump() for m in mcqs]}
    except QuizGenerationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/quiz/flashcards")
def generate_flashcards(request: FlashcardRequest) -> dict:
    try:
        cards = quiz_service.generate_flashcards(request.doc_id, num_cards=request.num_cards)
        return {"flashcards": [c.model_dump() for c in cards]}
    except QuizGenerationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
