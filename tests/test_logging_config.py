import json
import logging

import pytest

from app.logging_config import JsonFormatter, setup_logging


@pytest.fixture(autouse=True)
def restore_root():
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


def record(msg="hello %s", args=("world",), exc=False):
    info = None
    if exc:
        try:
            raise ValueError("boom")
        except ValueError:
            import sys
            info = sys.exc_info()
    return logging.LogRecord("app.x", logging.WARNING, __file__, 1, msg, args, info)


def test_json_line_has_fields():
    out = json.loads(JsonFormatter().format(record()))
    assert out["msg"] == "hello world" and out["level"] == "WARNING" and out["logger"] == "app.x"
    assert out["ts"].endswith("+00:00") and "exc" not in out


def test_json_includes_exception_and_unicode():
    out = json.loads(JsonFormatter().format(record("café ✓", (), exc=True)))
    assert out["msg"] == "café ✓" and "ValueError: boom" in out["exc"]


def test_setup_is_idempotent_and_replaces_its_own_handler():
    setup_logging("INFO", "text")
    setup_logging("DEBUG", "json")
    ours = [h for h in logging.getLogger().handlers if getattr(h, "_exam_qa_handler", False)]
    assert len(ours) == 1 and isinstance(ours[0].formatter, JsonFormatter)
    assert logging.getLogger().level == logging.DEBUG


def test_env_defaults_and_bad_level(monkeypatch):
    monkeypatch.setenv("LOG_FORMAT", "json")
    monkeypatch.setenv("LOG_LEVEL", "nonsense")
    setup_logging()
    assert logging.getLogger().level == logging.INFO
    assert any(isinstance(h.formatter, JsonFormatter) for h in logging.getLogger().handlers)
