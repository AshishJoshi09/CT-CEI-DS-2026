# 📚 AI Study Assistant (RAG)

An intelligent study assistant that lets students upload PDF, DOCX, and TXT
study material and ask questions about it in natural language. Answers are
**grounded in the uploaded material only** — the assistant clearly says so
when the answer isn't in the documents, instead of making things up.

Built with **LangChain + LangGraph + ChromaDB + OpenAI**, a **FastAPI**
backend, and a **Streamlit** chat UI.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [RAG Workflow](#rag-workflow)
- [Technologies Used](#technologies-used)
- [Folder Structure](#folder-structure)
- [Installation](#installation)
- [Environment Variables](#environment-variables)
- [How to Run](#how-to-run)
- [Example Questions](#example-questions)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)

---

## Features

- 📤 **Multi-format upload** — PDF, DOCX, TXT, multiple files at once
- 🔍 **Grounded Q&A** — answers come only from your documents; the assistant
  explicitly says "not found in the material" instead of hallucinating
- 📎 **Source citations** — every answer shows which file (and page, for PDFs)
  it came from
- 📄 **Document-specific search** — restrict a question to a single document,
  or search across all of them
- 🧠 **Chat memory** — follow-up questions like "give me an example" resolve
  correctly against the previous turn
- 📝 **Study tools** — short/detailed summaries, key points, definitions,
  formula extraction, MCQ quizzes, flashcards, important questions, revision
  notes, and "explain like I'm new to this" simplification
- ⚙️ **Configurable RAG settings** — chunk size, overlap, top-k, temperature,
  model name
- 🛡️ **Robust error handling** — invalid files, corrupted PDFs, oversized
  uploads, missing API keys, empty questions, and LLM/vector-store failures
  all produce friendly messages instead of crashes
- 🔌 **Optional REST API** (FastAPI) alongside the Streamlit UI

---

## Architecture

```
┌─────────────────┐        ┌──────────────────┐        ┌─────────────────┐
│  Streamlit UI    │◄──────►│  Service Layer    │◄──────►│  RAG Core        │
│  (chat, sidebar,  │        │  (qa/summary/quiz) │        │ (loader, chunker,│
│   study tools)    │        │                    │        │  embeddings,     │
└─────────────────┘        └──────────────────┘        │  vector_store,   │
        ▲                            ▲                    │  retriever,      │
        │                            │                    │  prompts, chain) │
        │                    ┌───────┴────────┐            └────────┬────────┘
        │                    │  LangGraph      │                     │
        │                    │  RAG workflow   │◄────────────────────┘
        │                    └───────┬────────┘
        │                            ▼
        │                    ┌──────────────┐         ┌──────────────┐
        └───────────────────►│  ChromaDB     │         │  OpenAI API   │
                              │  (vectors)    │         │  (LLM +      │
                              └──────────────┘         │  embeddings) │
                                                         └──────────────┘

FastAPI (app/main.py) exposes the same service layer over REST, for
programmatic access independent of the Streamlit UI.
```

The service layer (`app/services/`) is the single source of truth used by
**both** the Streamlit UI and the FastAPI backend, so there is no logic
duplication between the two frontends.

---

## RAG Workflow

Document ingestion:

```
Upload file → Extract text (loader.py) → Clean & split into chunks (chunker.py)
→ Generate embeddings (embeddings.py) → Store in ChromaDB with metadata
  (filename, doc_id, chunk_id, page)
```

Question answering (implemented as a **LangGraph** state machine in
`app/agents/graph.py`):

```
START
  │
  ▼
understand_query        — rewrite follow-up questions into standalone form
  │                        using chat history (so "give me an example" resolves
  │                        to the previous topic)
  ▼
retrieve_documents       — similarity search in ChromaDB (optionally scoped
  │                        to one document), top-k configurable
  ▼
check_context            — a cheap LLM call judges: is there enough context
  │                        to actually answer this question?
  │
  ├── insufficient ──► no_context_answer ──► END
  │                     (returns the standard "not found in the study
  │                      material" message, no hallucination risk)
  │
  └── sufficient ────► generate_answer ──► validate_answer ──► END
                        (grounded answer generated only from retrieved
                         chunks, then validated for a non-empty response)
```

LangGraph was used specifically because of the **conditional branch**
after context retrieval — this is exactly the kind of stateful branching
logic LangGraph is designed for, as opposed to a purely linear LCEL chain.

---

## Technologies Used

| Layer              | Technology                                   |
|--------------------|-----------------------------------------------|
| Backend API        | FastAPI, Uvicorn                              |
| Orchestration      | LangChain, LangGraph                          |
| LLM                | OpenAI (`gpt-4o-mini` by default)             |
| Embeddings         | OpenAI (`text-embedding-3-small` by default)  |
| Vector database    | ChromaDB (persistent, local)                  |
| Document parsing   | `pypdf` (via `PyPDFLoader`), `python-docx`    |
| Frontend           | Streamlit                                     |
| Validation/config  | Pydantic, pydantic-settings, python-dotenv    |
| Testing            | pytest                                        |

---

## Folder Structure

```
ai-study-assistant/
│
├── app/
│   ├── main.py                 # Optional FastAPI backend
│   ├── ui/
│   │   ├── sidebar.py          # Upload, doc management, settings
│   │   ├── chat.py             # Chat tab + study tools tab
│   │   └── components.py       # Reusable Streamlit widgets
│   │
│   ├── rag/
│   │   ├── loader.py           # PDF/DOCX/TXT text extraction
│   │   ├── chunker.py          # Text splitting + chunk metadata
│   │   ├── embeddings.py       # Embedding model factory
│   │   ├── vector_store.py     # ChromaDB wrapper
│   │   ├── retriever.py        # Similarity search + context formatting
│   │   ├── prompts.py          # All prompt templates
│   │   └── chain.py            # LLM factory + LCEL chains
│   │
│   ├── agents/
│   │   └── graph.py            # LangGraph RAG workflow
│   │
│   ├── services/
│   │   ├── qa_service.py       # Ingestion + Q&A + chat memory orchestration
│   │   ├── summary_service.py  # Summaries / key points / definitions
│   │   └── quiz_service.py     # MCQs / flashcards / revision notes
│   │
│   ├── utils/
│   │   ├── config.py           # Settings (env-driven)
│   │   └── helpers.py          # Upload validation, formatting
│   │
│   └── models/
│       └── schemas.py          # Pydantic request/response models
│
├── data/
│   ├── uploads/                # Saved uploaded files (gitignored)
│   └── chroma/                 # Persistent vector DB (gitignored)
│
├── tests/
│   ├── test_loader.py
│   ├── test_chunking.py
│   └── test_rag.py
│
├── streamlit_app.py            # Streamlit entry point
├── run.py                      # `python run.py` launcher
├── .env / .env.example
├── .gitignore
├── requirements.txt
├── pytest.ini
└── README.md
```

---

## Installation

**Prerequisites:** Python 3.11+, an OpenAI API key.

```bash
# 1. Clone / unzip the project, then enter it
cd ai-study-assistant

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Environment Variables

Copy the example file and add your key:

```bash
cp .env.example .env
```

`.env`:
```
OPENAI_API_KEY=your_openai_api_key_here

# Optional overrides — sensible defaults are used if omitted
LLM_MODEL_NAME=gpt-4o-mini
EMBEDDING_MODEL_NAME=text-embedding-3-small
TEMPERATURE=0.2
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K=4
MAX_FILE_SIZE_MB=25
```

The app will refuse to run (with a clear on-screen message) if no valid
`OPENAI_API_KEY` is set.

---

## How to Run

### Streamlit UI (primary app)

```bash
streamlit run streamlit_app.py
# or:
python run.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

### FastAPI backend (optional)

```bash
uvicorn app.main:app --reload
```

API docs available at `http://localhost:8000/docs`.

---

## Example Questions

Once you've uploaded some notes, try:

- "What is supervised learning?"
- "Explain the difference between classification and regression."
- "What are the advantages of neural networks?"
- "Summarize the chapter about operating systems."
- "Explain deadlock." *(searches across all uploaded documents)*
- "What is overfitting?" *(scoped to a single selected document)*
- Follow-up: "Give me an example." *(chat memory resolves "it"/"that")*

---

## Testing

```bash
pytest
```

- `test_loader.py` and `test_chunking.py` run fully offline (no API key
  needed) — they test text extraction and chunk metadata directly.
- `test_rag.py` contains both offline unit tests (mocked vector store) and
  integration tests marked to **auto-skip** unless `OPENAI_API_KEY` is set,
  since they exercise real embeddings/LLM calls end-to-end (upload → ask →
  citation check, multi-document scoping, "not found" behavior).

---

## Troubleshooting

| Problem | Likely cause / fix |
|---|---|
| "No OpenAI API key found" on startup | Set `OPENAI_API_KEY` in `.env` and restart the app. |
| "Could not read PDF... corrupted or password-protected" | The PDF is scanned (image-only) or encrypted — OCR isn't included in this project; try a text-based PDF. |
| "File is too large" | Increase `MAX_FILE_SIZE_MB` in `.env`, or split the document. |
| Answers seem to ignore recently uploaded content | Make sure the upload finished (green success message) before asking — embedding large files takes a few seconds. |
| `ModuleNotFoundError` when running tests | Run `pytest` from the project root so `pytest.ini`'s `pythonpath = .` takes effect. |
| ChromaDB errors on first run | Ensure `data/chroma/` is writable; delete it and restart to reset the vector database. |
| Slow responses | Lower `top_k`, or use a smaller/faster model via `LLM_MODEL_NAME`. |

---

## Future Improvements

- OCR support for scanned/image-based PDFs
- Persistent chat history across sessions (currently session-only)
- User accounts / multi-tenant document isolation
- Streaming token-by-token answers in the UI
- Export quizzes/flashcards/notes to PDF or Anki format
- Support for additional file types (PPTX, Markdown, web URLs)
- Swap-in support for local/open-source LLMs and embedding models
