"""PDF text extraction and sentence-aware chunking."""
from __future__ import annotations

import re
from pathlib import Path
from typing import BinaryIO, List, Union

from pypdf import PdfReader


class EmptyPDFError(ValueError):
    """The PDF has no extractable text (e.g. a scanned document)."""


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """Split text into ~chunk_size-word chunks on sentence boundaries.

    Consecutive chunks share roughly `overlap` words so context isn't lost.
    """
    sentences = re.split(r"(?<=[.!?])\s+", _clean(text))
    sentences = [s for s in sentences if s]
    chunks: List[str] = []
    current: List[str] = []
    count = 0
    for sent in sentences:
        n = len(sent.split())
        if current and count + n > chunk_size:
            chunks.append(" ".join(current))
            # carry trailing sentences worth ~overlap words into the next chunk
            carry: List[str] = []
            carried = 0
            for s in reversed(current):
                w = len(s.split())
                if carried + w > overlap:
                    break
                carry.insert(0, s)
                carried += w
            current, count = carry, carried
        current.append(sent)
        count += n
    if current:
        chunks.append(" ".join(current))
    return chunks


def load_pdf(source: Union[str, Path, BinaryIO], *, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """Read a PDF (path or file-like) and return text chunks.

    Raises EmptyPDFError if no text could be extracted.
    """
    reader = PdfReader(source if hasattr(source, "read") else str(source))
    text = " ".join(page.extract_text() or "" for page in reader.pages)
    chunks = chunk_text(text, chunk_size, overlap)
    if not chunks:
        raise EmptyPDFError(
            "No text found in this PDF. It may be a scanned document without a text layer."
        )
    return chunks
