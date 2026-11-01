"""PDF text extraction and sentence-aware chunking."""
from __future__ import annotations

import io
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


def chunks_from_pages(pages: List[str], *, chunk_size: int = 300, overlap: int = 50,
                      mode: Optional[str] = None, paged: bool = True) -> List[Chunk]:
    """Chunk already-extracted page texts (possibly returning an empty list).

    Legacy mode (default): chunks never span pages, so every chunk can be cited as "file, p.N".
    Structured mode (CHUNKING=structured): chunks follow headings and may span pages; `page` is the
    first page and `page_end` the last when different.
    With paged=False (Word, Markdown, text, web pages) there is no real page number: chunks get page 0,
    which the app treats as "no page"."""
    if (mode or config.CHUNKING) == "structured":
        chunks = [Chunk(c["text"], c["page"], c["heading"], c["page_end"] if c["page_end"] != c["page"] else None)
                  for c in structured_chunks(pages, config.CHUNK_TARGET_TOKENS, config.CHUNK_OVERLAP_TOKENS)]
    else:
        chunks = [Chunk(text, page_no) for page_no, page_text in enumerate(pages, 1)
                  for text in chunk_text(page_text or "", chunk_size, overlap)]
    if not paged:
        chunks = [c._replace(page=0, page_end=None) for c in chunks]
    return chunks


_NO_TEXT = "No text found in this PDF. It may be a scanned document without a text layer."


def load_pdf(source: Union[str, Path, BinaryIO], *, chunk_size: int = 300, overlap: int = 50,
             mode: Optional[str] = None, ocr=None) -> List[Chunk]:
    """Read a PDF (path or file-like) and return page-tagged text chunks (see `chunks_from_pages`).

    If the PDF has no text layer and OCR is enabled (OCR=auto and an OCR backend is installed, or an `ocr`
    callable `ocr(pdf_bytes) -> list[str]` is given), the pages are recognised from images instead.
    Raises EmptyPDFError if no text could be obtained.
    """
    raw = source.read() if hasattr(source, "read") else Path(source).read_bytes()
    reader = PdfReader(io.BytesIO(raw))
    pages = [page.extract_text() or "" for page in reader.pages]
    chunks = chunks_from_pages(pages, chunk_size=chunk_size, overlap=overlap, mode=mode)
    if not chunks:
        from .ocr import ocr_pdf  # lazy: optional dependencies

        recognised = (ocr or ocr_pdf)(raw)
        chunks = chunks_from_pages(recognised, chunk_size=chunk_size, overlap=overlap, mode=mode)
    if not chunks:
        raise EmptyPDFError(_NO_TEXT)
    return chunks
