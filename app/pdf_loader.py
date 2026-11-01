"""PDF text extraction and sentence-aware chunking."""
from __future__ import annotations

import re
from pathlib import Path
from typing import BinaryIO, List, NamedTuple, Optional, Union

from pypdf import PdfReader

from . import config
from .chunking import structured_chunks


class Chunk(NamedTuple):
    text: str
    page: int  # 1-based page number the chunk starts on
    heading: str = ""  # section path, e.g. "2 Cells > 2.1 Mitochondria" (structured chunking only)
    page_end: Optional[int] = None  # last page, if the chunk spans pages


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


def load_pdf(source: Union[str, Path, BinaryIO], *, chunk_size: int = 300, overlap: int = 50,
             mode: Optional[str] = None) -> List[Chunk]:
    """Read a PDF (path or file-like) and return page-tagged text chunks.

    Legacy mode (default): chunks never span pages, so every chunk can be cited as "file, p.N".
    Structured mode (CHUNKING=structured): chunks follow headings and may span pages; `page` is the
    first page and `page_end` the last when different.
    Raises EmptyPDFError if no text could be extracted.
    """
    reader = PdfReader(source if hasattr(source, "read") else str(source))
    if (mode or config.CHUNKING) == "structured":
        pages = [page.extract_text() or "" for page in reader.pages]
        chunks = [Chunk(c["text"], c["page"], c["heading"], c["page_end"] if c["page_end"] != c["page"] else None)
                  for c in structured_chunks(pages, config.CHUNK_TARGET_TOKENS, config.CHUNK_OVERLAP_TOKENS)]
        if not chunks:
            raise EmptyPDFError(
                "No text found in this PDF. It may be a scanned document without a text layer."
            )
        return chunks
    chunks = []
    for page_no, page in enumerate(reader.pages, 1):
        for text in chunk_text(page.extract_text() or "", chunk_size, overlap):
            chunks.append(Chunk(text, page_no))
    if not chunks:
        raise EmptyPDFError(
            "No text found in this PDF. It may be a scanned document without a text layer."
        )
    return chunks
