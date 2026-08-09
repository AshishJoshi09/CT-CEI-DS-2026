"""
Factory for the embedding model used to vectorize document chunks and queries.
Centralized here so swapping providers only requires changing this file.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_openai import OpenAIEmbeddings

from app.utils.config import settings


@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    """
    Return a cached OpenAIEmbeddings instance.
    Cached with lru_cache since embedding clients are safe to reuse and
    re-instantiating per call would add overhead.
    """
    return OpenAIEmbeddings(
        model=settings.embedding_model_name,
        api_key=settings.openai_api_key,
    )
