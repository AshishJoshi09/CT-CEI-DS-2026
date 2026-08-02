"""
Stage 3: Embedding Creation
Converts text (chunks at ingestion time, queries at query time) into vector
representations capturing semantic meaning. Uses sentence-transformers, which
runs locally (no API key/cost).
"""

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str):
        print(f"Loading embedding model '{model_name}' ...")
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Batch-embeds a list of strings. Used for chunks at index time."""
        return self.model.encode(
            texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True
        )

    def embed_query(self, query: str) -> np.ndarray:
        """Embeds a single query string."""
        return self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )[0]
