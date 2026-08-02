"""
Stage 1: Document Ingestion
Loads PDFs / .txt files and converts them into raw text, tagged with source metadata
so answers can later be traced back to the file (and page, for PDFs) they came from.
"""

import os
from dataclasses import dataclass


@dataclass
class RawDocument:
    text: str
    source: str      # filename
    page: int = None # page number for PDFs, None for plain text


def load_pdf(filepath: str) -> list[RawDocument]:
    from pypdf import PdfReader

    reader = PdfReader(filepath)
    docs = []
    filename = os.path.basename(filepath)

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            docs.append(RawDocument(text=text, source=filename, page=page_num))
    return docs


def load_txt(filepath: str) -> list[RawDocument]:
    filename = os.path.basename(filepath)
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().strip()
    return [RawDocument(text=text, source=filename)] if text else []


def load_documents(directory: str) -> list[RawDocument]:
    """Loads every .pdf / .txt file in a directory."""
    if not os.path.isdir(directory):
        raise FileNotFoundError(
            f"'{directory}' not found. Create it and drop your PDFs/.txt files inside."
        )

    all_docs = []
    for filename in sorted(os.listdir(directory)):
        filepath = os.path.join(directory, filename)
        ext = filename.lower().split(".")[-1]

        if ext == "pdf":
            all_docs.extend(load_pdf(filepath))
        elif ext == "txt":
            all_docs.extend(load_txt(filepath))
        else:
            continue  # skip unsupported file types

    if not all_docs:
        raise ValueError(f"No readable PDF/.txt files found in '{directory}'.")

    print(f"Loaded {len(all_docs)} page(s)/document(s) from '{directory}'.")
    return all_docs
