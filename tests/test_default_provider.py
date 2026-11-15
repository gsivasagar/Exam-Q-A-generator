"""Which provider the Streamlit sidebar preselects, and the upload limit the host shows."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app import config, llm

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("setting, reachable, expected", [
    ("auto", True, "Ollama"),
    ("auto", False, "Gemini"),
    ("ollama", False, "Ollama"),
    ("gemini", True, "Gemini"),
])
def test_default_provider_choice(monkeypatch, setting, reachable, expected):
    monkeypatch.setattr(config, "DEFAULT_PROVIDER", setting)
    monkeypatch.setattr(llm, "ollama_reachable", lambda *a, **k: reachable)
    assert llm.default_provider() == expected


def test_ollama_reachable_is_false_when_nothing_listens():
    assert llm.ollama_reachable("http://127.0.0.1:9", timeout=0.2) is False


def _open_app():
    return AppTest.from_file(str(ROOT / "app" / "ui.py"), default_timeout=60)


def test_sidebar_starts_on_gemini_when_ollama_is_missing(store, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "q.db")
    monkeypatch.setattr(config, "DEFAULT_PROVIDER", "auto")
    monkeypatch.setattr(llm, "ollama_reachable", lambda *a, **k: False)
    at = _open_app().run()
    assert not at.exception
    assert next(r for r in at.radio if r.label == "Select LLM Provider").value == "Gemini"
    assert any(t.label == "Gemini API Key" for t in at.text_input)


def test_sidebar_starts_on_ollama_when_it_answers(store, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "q.db")
    monkeypatch.setattr(config, "DEFAULT_PROVIDER", "auto")
    monkeypatch.setattr(llm, "ollama_reachable", lambda *a, **k: True)
    at = _open_app().run()
    assert next(r for r in at.radio if r.label == "Select LLM Provider").value == "Ollama"


def test_upload_box_limit_matches_the_app_limit():
    text = (ROOT / ".streamlit" / "config.toml").read_text()
    assert f"maxUploadSize = {int(config.MAX_UPLOAD_MB)}" in text
