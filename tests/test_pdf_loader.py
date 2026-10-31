import io

import pytest

from app.pdf_loader import EmptyPDFError, chunk_text, load_pdf
from tests.conftest import make_pdf_bytes


def test_chunks_respect_size_and_sentences():
    text = " ".join(f"This is sentence number {i} of the document." for i in range(200))
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    assert len(chunks) > 5
    for c in chunks:
        assert len(c.split()) <= 60          # roughly bounded
        assert c.endswith(".")               # never cut mid-sentence


def test_chunks_overlap():
    text = " ".join(f"Sentence {i} is here now." for i in range(100))
    chunks = chunk_text(text, chunk_size=40, overlap=10)
    for prev, nxt in zip(chunks, chunks[1:]):
        last_sentence = prev.split(". ")[-1]
        assert last_sentence in nxt   # the tail of the previous chunk is repeated


def test_empty_text_gives_no_chunks():
    assert chunk_text("   ") == []


def test_load_pdf_from_bytes():
    chunks = load_pdf(io.BytesIO(make_pdf_bytes(["Cells are the unit of life."] * 30)))
    assert chunks and "Cells" in chunks[0]


def test_pdf_without_text_raises():
    from fpdf import FPDF
    pdf = FPDF(); pdf.add_page()
    with pytest.raises(EmptyPDFError):
        load_pdf(io.BytesIO(bytes(pdf.output())))
