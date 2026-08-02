"""
Stage 4: Vector Database
Stores chunk embeddings in a FAISS index for fast similarity search, and persists
the index + chunk metadata to disk so you don't need to re-embed on every run.
"""

import os
import pickle
import numpy as np
import faiss

from chunker import Chunk


class VectorStore:
    def __init__(self, dim: int = None):
        self.index = faiss.IndexFlatIP(dim) if dim else None  # cosine sim via inner product on normalized vecs
        self.chunks: list[Chunk] = []

    def build(self, embeddings: np.ndarray, chunks: list[Chunk]):
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings.astype("float32"))
        self.chunks = chunks
        print(f"Vector store built with {self.index.ntotal} vectors.")

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[Chunk, float]]:
        query_embedding = query_embedding.astype("float32").reshape(1, -1)
        scores, indices = self.index.search(query_embedding, top_k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)
        print(f"Vector store saved to '{path}'.")

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        store = cls()
        store.index = faiss.read_index(os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "chunks.pkl"), "rb") as f:
            store.chunks = pickle.load(f)
        print(f"Vector store loaded from '{path}' ({store.index.ntotal} vectors).")
        return store

    @staticmethod
    def exists(path: str) -> bool:
        return os.path.exists(os.path.join(path, "index.faiss"))
