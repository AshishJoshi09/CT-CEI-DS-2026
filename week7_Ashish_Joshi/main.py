"""
Entry point.

First run  : reads documents/, chunks + embeds them, builds & saves the vector store.
Later runs : loads the saved vector store directly (fast, no re-embedding).

Usage:
    python main.py                       # interactive Q&A loop
    python main.py --rebuild             # force re-ingesting documents/
    python main.py --ask "your question" # single question, non-interactive
"""

import argparse
import os

import config
from document_loader import load_documents
from chunker import chunk_documents
from embedder import Embedder
from vector_store import VectorStore
from rag_pipeline import RAGPipeline


def build_vector_store(embedder: Embedder) -> VectorStore:
    raw_docs = load_documents(config.DOCUMENTS_DIR)
    chunks = chunk_documents(raw_docs, config.CHUNK_SIZE, config.CHUNK_OVERLAP)

    texts = [c.text for c in chunks]
    embeddings = embedder.embed_texts(texts)

    store = VectorStore()
    store.build(embeddings, chunks)
    store.save(config.VECTOR_STORE_PATH)
    return store


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Re-ingest documents/ from scratch")
    parser.add_argument("--ask", type=str, default=None, help="Ask a single question and exit")
    args = parser.parse_args()

    if not config.GROQ_API_KEY:
        if os.name == "nt":
            raise EnvironmentError(
                "GROQ_API_KEY is not set.\n"
                "Windows PowerShell: $env:GROQ_API_KEY='gsk_...'\n"
                "Windows Command Prompt: set GROQ_API_KEY=gsk_...\n"
                "Permanent for new cmd windows: setx GROQ_API_KEY gsk_..."
            )

        raise EnvironmentError(
            "GROQ_API_KEY is not set. Run: export GROQ_API_KEY='gsk_...'"
        )

    embedder = Embedder(config.EMBEDDING_MODEL)

    if args.rebuild or not VectorStore.exists(config.VECTOR_STORE_PATH):
        store = build_vector_store(embedder)
    else:
        store = VectorStore.load(config.VECTOR_STORE_PATH)

    pipeline = RAGPipeline(store, embedder)

    if args.ask:
        print("\n" + pipeline.answer(args.ask, verbose=True))
        return

    print("\nRAG system ready. Ask a question (type 'exit' to quit).\n")
    while True:
        query = input("You: ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue
        answer = pipeline.answer(query, verbose=True)
        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    main()
