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
        def generate(self, prompt, *, schema=None, **_):
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


def test_mcq_flow_and_weak_topic_practice(store, tmp_path, monkeypatch):
    import json
    import streamlit as st
    from app import config, llm

    monkeypatch.setattr(config, "DB_PATH", tmp_path / "q.db")
    seen_prompts = []

    class McqLLM(FakeLLM):
        def generate(self, prompt, *, schema=None, **_):
            seen_prompts.append(prompt)
            self.calls += 1
            qs = [{"question": f"Q{self.calls}-{i}?", "distractors": ["bad1", "bad2", "bad3"],
                   "answer": "right", "topic": "Calvin" if i % 2 else "Glycolysis", "chunk": 1}
                  for i in range(5)]
            return json.dumps({"questions": qs})

    monkeypatch.setattr(llm, "get_provider", lambda *a, **k: McqLLM())

    class Up:
        name = "bio.pdf"
        def getvalue(self): return make_pdf_bytes([f"Glycolysis fact {i}." for i in range(60)])
    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: [Up()])

    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app" / "ui.py"), default_timeout=60)
    at.run()
    next(r for r in at.radio if r.label == "Question type").set_value("Multiple choice")
    next(b for b in at.button if b.label == "Create Quiz").click().run()
    assert not at.exception and not at.error, (at.exception, [e.value for e in at.error])

    qa = at.session_state["qa_pairs"]
    assert len(qa) == 10 and all(q["type"] == "mcq" for q in qa)
    assert all(q["source"] == "bio.pdf" and q["page"] == 1 for q in qa)   # citations flowed through

    # answer: first 3 correctly, rest deliberately wrong
    quiz_radios = [r for r in at.radio if r.label == "Choose one"]
    assert len(quiz_radios) == 10
    for i, r in enumerate(quiz_radios):
        r.set_value(qa[i]["answer"] if i < 3 else next(o for o in qa[i]["options"] if o != qa[i]["answer"]))
    next(b for b in at.button if "grade me" in b.label).click().run()
    assert not at.exception, at.exception
    graded = at.session_state["graded"]
    assert [g["score"] for g in graded] == [1.0] * 3 + [0.0] * 7

    # weak-topic practice builds a new quiz focused on the weak topics
    seen_prompts.clear()
    old_quiz = at.session_state["quiz_id"]
    next(b for b in at.button if "Practice my weak topics" in b.label).click().run()
    assert not at.exception and not at.error, (at.exception, [e.value for e in at.error])
    assert at.session_state["quiz_id"] != old_quiz
    assert "graded" not in at.session_state
    assert any("Glycolysis" in p and "Calvin" in p for p in seen_prompts)
    assert len([r for r in at.radio if r.label == "Choose one"]) == 10


def test_retry_ungraded_answers(store, tmp_path, monkeypatch):
    import streamlit as st
    from app import config, llm
    from app.llm import LLMError

    monkeypatch.setattr(config, "DB_PATH", tmp_path / "q.db")
    state = {"fail": True}

    class FlakyGrader(FakeLLM):
        def generate(self, prompt, *, schema=None, **_):
            if "Evaluate the student" in prompt:
                if state["fail"]:
                    raise LLMError("grader down")
                return '{"score": 0.8, "feedback": "Recovered"}'
            return super().generate(prompt, schema=schema)

    class Up:
        name = "notes.pdf"
        def getvalue(self): return make_pdf_bytes([f"Cells divide by mitosis {i}." for i in range(60)])

    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: [Up()])
    monkeypatch.setattr(llm, "get_provider", lambda *a, **k: FlakyGrader())
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app" / "ui.py"), default_timeout=60)
    at.run()
    next(b for b in at.button if b.label == "Create Quiz").click().run()
    for ta in at.text_area:
        ta.input("a real attempt at the answer")
    next(b for b in at.button if "grade me" in b.label).click().run()
    assert not at.exception, at.exception
    assert all(g["score"] is None for g in at.session_state["graded"])

    state["fail"] = False
    next(b for b in at.button if "Retry ungraded" in b.label).click().run()
    assert not at.exception, at.exception
    assert all(g["score"] == 0.8 and not g.get("error") for g in at.session_state["graded"])
    assert not any("Retry ungraded" in b.label for b in at.button)  # nothing left to retry


def test_app_starts_without_pythonpath_or_install(tmp_path):
    """Regression: Streamlit Cloud runs `streamlit run app/ui.py` with nothing installed and no PYTHONPATH,
    which failed with "No module named 'app'". -I ignores PYTHONPATH and the current directory."""
    import subprocess
    import sys
    root = Path(__file__).resolve().parent.parent
    code = ("from streamlit.testing.v1 import AppTest\n"
            f"at = AppTest.from_file({str(root / 'app' / 'ui.py')!r}, default_timeout=60)\n"
            "at.run()\n"
            "print('EXC', [str(e.value)[:80] for e in at.exception])\n")
    r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, cwd=tmp_path,
                       env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "DATA_DIR": str(tmp_path / "data")})
    assert "EXC []" in r.stdout, (r.stdout + r.stderr)[-600:]
