"""
Small, dependency-light helper functions used across the app:
file validation, safe saving of uploads, and text/source formatting.
"""
from __future__ import annotations

import re
import uuid
from pathlib import Path

from app.utils.config import settings


class UploadValidationError(Exception):
    """Raised when an uploaded file fails validation."""


def get_file_extension(filename: str) -> str:
    """Return the lowercase extension of a filename, including the dot."""
    return Path(filename).suffix.lower()


def validate_upload(filename: str, size_bytes: int) -> None:
    """
    Validate an uploaded file's extension and size before it touches disk.
    Raises UploadValidationError with a user-friendly message on failure.
    """
    ext = get_file_extension(filename)
    if ext not in settings.allowed_extensions:
        allowed = ", ".join(settings.allowed_extensions)
        raise UploadValidationError(
            f"Unsupported file type '{ext}'. Allowed types: {allowed}"
        )

    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if size_bytes <= 0:
        raise UploadValidationError("The uploaded file appears to be empty.")
    if size_bytes > max_bytes:
        raise UploadValidationError(
            f"File is too large ({size_bytes / (1024 * 1024):.1f} MB). "
            f"Maximum allowed size is {settings.max_file_size_mb} MB."
        )


def safe_filename(filename: str) -> str:
    """Sanitize a filename and prefix it with a short unique id to avoid collisions."""
    stem = Path(filename).stem
    ext = Path(filename).suffix
    stem = re.sub(r"[^a-zA-Z0-9_\-]", "_", stem)[:80]
    unique = uuid.uuid4().hex[:8]
    return f"{stem}_{unique}{ext}"


def save_uploaded_file(file_bytes: bytes, filename: str) -> Path:
    """
    Persist raw uploaded bytes to the uploads directory.
    Does NOT execute or parse the file - just writes bytes to disk.
    """
    validate_upload(filename, len(file_bytes))
    target_name = safe_filename(filename)
    target_path = settings.upload_dir / target_name
    target_path.write_bytes(file_bytes)
    return target_path


def truncate(text: str, max_chars: int = 220) -> str:
    """Truncate text for display in UI snippets."""
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def format_bytes(size_bytes: int) -> str:
    """Human-readable byte size, e.g. '1.4 MB'."""
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"
