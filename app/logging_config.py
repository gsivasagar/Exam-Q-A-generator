"""Logging setup: readable text by default, one JSON object per line with LOG_FORMAT=json."""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Optional

_MARK = "_exam_qa_handler"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)


def setup_logging(level: Optional[str] = None, fmt: Optional[str] = None) -> None:
    """Configure the root logger once. Safe to call on every Streamlit rerun (it replaces its own
    handler instead of adding another). LOG_LEVEL (default INFO) and LOG_FORMAT (text|json)."""
    level_name = str(level or os.environ.get("LOG_LEVEL") or "INFO").upper()
    fmt_name = str(fmt or os.environ.get("LOG_FORMAT") or "text").lower()
    root = logging.getLogger()
    for h in list(root.handlers):
        if getattr(h, _MARK, False):
            root.removeHandler(h)
    handler = logging.StreamHandler(sys.stderr)
    setattr(handler, _MARK, True)
    handler.setFormatter(JsonFormatter() if fmt_name == "json"
                         else logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)
    root.setLevel(getattr(logging, level_name, logging.INFO))
