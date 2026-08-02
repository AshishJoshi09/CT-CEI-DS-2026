"""
Central configuration for the RAG pipeline.

"""

import os

# ---- Document ingestion ----
DOCUMENTS_DIR = "documents"          

# ---- Chunking ----
CHUNK_SIZE = 800                     # characters per chunk
CHUNK_OVERLAP = 150                  # overlap between consecutive chunks

# ---- Embeddings ----
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ---- Vector store ----
VECTOR_STORE_PATH = "vector_store"   # where the FAISS index + metadata get persisted

# ---- Retrieval ----
TOP_K = 4                            # how many chunks to retrieve per query

# ---- Generation ----
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GENERATION_MODEL = "llama-3.3-70b-versatile"   # or "llama-3.1-8b-instant" for faster/lighter
MAX_TOKENS = 1024
