"""SQLite persistence of graded results, scoped by session."""
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd

from . import config


def _connect(path=None):
    p = path or config.DB_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(p)


def init_db(path=None) -> None:
    with closing(_connect(path)) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, session_id TEXT, quiz_id TEXT,
            question TEXT, topic TEXT, student TEXT, answer TEXT,
            score REAL, feedback TEXT)""")
        conn.commit()


def store_results(graded: List[dict], session_id: str, quiz_id: str, path=None) -> None:
    """Persist graded answers. Ungradable answers (score None) are skipped."""
    init_db(path)
    now = datetime.now(timezone.utc).isoformat()
    rows = [(now, session_id, quiz_id, g["question"], g.get("topic"), g.get("student", ""),
             g["answer"], g["score"], g.get("feedback", ""))
            for g in graded if g.get("score") is not None]
    with closing(_connect(path)) as conn:
        conn.executemany(
            "INSERT INTO results (timestamp, session_id, quiz_id, question, topic, student,"
            " answer, score, feedback) VALUES (?,?,?,?,?,?,?,?,?)", rows)
        conn.commit()


def load_results(session_id: Optional[str] = None, path=None) -> pd.DataFrame:
    init_db(path)
    with closing(_connect(path)) as conn:
        if session_id:
            return pd.read_sql_query("SELECT * FROM results WHERE session_id = ?", conn, params=(session_id,))
        return pd.read_sql_query("SELECT * FROM results", conn)
