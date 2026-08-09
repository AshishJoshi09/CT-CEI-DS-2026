"""
Centralized application configuration.

All environment-driven settings live here so the rest of the codebase
never touches os.environ directly. Uses pydantic-settings for validation
and sensible typed defaults.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = two levels up from this file (app/utils/config.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application settings, populated from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LLM / OpenAI ---
    openai_api_key: str = Field(default="", description="OpenAI API key")
    llm_model_name: str = Field(default="gpt-4o-mini", description="Chat model used for generation")
    embedding_model_name: str = Field(
        default="text-embedding-3-small", description="Embedding model used for vectorization"
    )
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)

    # --- RAG / chunking ---
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)
    top_k: int = Field(default=4, gt=0, le=20)

    # --- Storage paths ---
    upload_dir: Path = Field(default=PROJECT_ROOT / "data" / "uploads")
    chroma_dir: Path = Field(default=PROJECT_ROOT / "data" / "chroma")

    # --- Upload limits / security ---
    max_file_size_mb: int = Field(default=25, gt=0)
    allowed_extensions: tuple[str, ...] = (".pdf", ".docx", ".txt")

    # --- Collection name for the vector store ---
    chroma_collection_name: str = Field(default="study_assistant")

    def ensure_dirs(self) -> None:
        """Create storage directories if they don't already exist."""
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)


# Singleton settings instance used across the app
settings = Settings()
settings.ensure_dirs()


def has_valid_api_key() -> bool:
    """Return True if an OpenAI API key looks present (not empty/placeholder)."""
    key = settings.openai_api_key.strip()
    return bool(key) and key.lower() not in {"your_api_key_here", "changeme"}
