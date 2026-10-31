"""End-to-end UI flow with Streamlit's AppTest, a fake LLM and real Chroma/PDF code."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from tests.conftest import FakeLLM, make_pdf_bytes


@pytest.fixture
def app(store, tmp_path, monkeypatch):
    from app import config, llm
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "q.db")

    class GradingLLM(FakeLLM):
        def generate(self, prompt, *, schema=None):
            if "Evaluate the student" in prompt:
                return '{"score": 0.9, "feedback": "Well done"}'
            return super().generate(prompt, schema=schema)

    monkeypatch.setattr(llm, "get_provider", lambda *a, **k: GradingLLM())
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app" / "ui.py"), default_timeout=60)
    return at


def test_page_loads(app):
    app.run()
    assert not app.exception
    assert app.title[0].value.endswith("Exam Q&A Generator")


def test_full_quiz_flow(app, monkeypatch):
    # AppTest has no file_uploader support in all versions, so feed one in directly
    class Up:
        name = "notes.pdf"
        def getvalue(self): return make_pdf_bytes([f"Cells divide by mitosis {i}." for i in range(60)])
    import streamlit as st
    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: [Up()])

    app.run()
    next(b for b in app.button if b.label == "Create Quiz").click().run()
    assert not app.exception, app.exception
    assert not app.error, [e.value for e in app.error]
    assert len(app.session_state["qa_pairs"]) == 10

    for ta in app.text_area:
        ta.input("my answer")
    next(b for b in app.button if "grade me" in b.label).click().run()
    assert not app.exception, app.exception
    assert "graded" in app.session_state
    assert all(g["score"] == 0.9 for g in app.session_state["graded"])
    assert any("Average Score" == m.label for m in app.metric)
