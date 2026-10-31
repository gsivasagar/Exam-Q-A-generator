import json

import pytest
from fpdf import FPDF


class FakeLLM:
    """Deterministic stand-in for an LLM provider."""

    def __init__(self, replies=None):
        self.replies = list(replies or [])
        self.prompts = []
        self.calls = 0

    def generate(self, prompt, *, schema=None):
        self.prompts.append(prompt)
        self.calls += 1
        if self.replies:
            r = self.replies.pop(0)
            if isinstance(r, Exception):
                raise r
            return r
        # default: a fresh batch of unique questions each call
        base = self.calls * 100
        return json.dumps([{"question": f"Question {base + i}?", "answer": f"Answer {base + i}",
                            "topic": f"Topic{i % 3}"} for i in range(5)])


def make_pdf_bytes(sentences):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 6, " ".join(sentences))
    return bytes(pdf.output())


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def pdf_bytes():
    return make_pdf_bytes([f"Mitochondria produce energy number {i} for the cell." for i in range(60)])


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Real Chroma store in a temp dir."""
    from app import config, vector_store
    monkeypatch.setattr(config, "CHROMA_DIR", str(tmp_path / "chroma"))
    vector_store._collection.cache_clear()
    yield vector_store
    vector_store._collection.cache_clear()
