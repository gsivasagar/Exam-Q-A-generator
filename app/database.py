"""SQLite persistence of graded results, scoped by session."""
import sqlite3
import threading
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd

from . import config


def _connect(path=None):
    p = path or config.DB_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(p)


SCHEMA_VERSION = 3


def migrate(conn: sqlite3.Connection) -> int:
    """Bring the database up to SCHEMA_VERSION, one numbered step at a time (tracked with PRAGMA
    user_version). Safe to run repeatedly, and on databases created before versioning existed."""
    version = start = conn.execute("PRAGMA user_version").fetchone()[0]
    if version >= SCHEMA_VERSION:
        return version  # already current: no write, so readers never take the write lock for nothing
    if version < 1:  # original results table
        conn.execute("""CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, session_id TEXT, quiz_id TEXT,
            question TEXT, topic TEXT, student TEXT, answer TEXT,
            score REAL, feedback TEXT)""")
        version = 1
    if version < 2:  # saved quizzes, spaced repetition, document registry
        conn.execute("""CREATE TABLE IF NOT EXISTS quizzes (
            id TEXT PRIMARY KEY, owner TEXT NOT NULL, created_at TEXT NOT NULL, title TEXT,
            question_type TEXT, difficulty TEXT, n INTEGER, questions_json TEXT NOT NULL)""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_quizzes_owner ON quizzes(owner, created_at)")
        conn.execute("""CREATE TABLE IF NOT EXISTS srs (
            owner TEXT NOT NULL, qhash TEXT NOT NULL, question_json TEXT NOT NULL, due_at TEXT NOT NULL,
            interval_days REAL NOT NULL, ease REAL NOT NULL, reps INTEGER NOT NULL, last_score REAL,
            PRIMARY KEY (owner, qhash))""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_srs_due ON srs(owner, due_at)")
        conn.execute("CREATE TABLE IF NOT EXISTS docs (doc_id TEXT PRIMARY KEY, created_at TEXT NOT NULL)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_results_session ON results(session_id)")
        version = 2
    if version < 3:  # accounts, login sessions, per-user documents, graded attempts
        conn.execute("""CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY, email TEXT UNIQUE, password_hash TEXT,
            is_guest INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL, expires_at TEXT)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            token_hash TEXT NOT NULL UNIQUE, prev_hash TEXT, rotated_at TEXT, user_agent TEXT,
            remember INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL, last_used_at TEXT NOT NULL,
            expires_at TEXT NOT NULL, revoked_at TEXT)""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_prev ON sessions(prev_hash)")
        conn.execute("""CREATE TABLE IF NOT EXISTS attempts (
            id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, quiz_id TEXT NOT NULL, created_at TEXT NOT NULL,
            result_json TEXT NOT NULL)""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_attempts_quiz ON attempts(owner_id, quiz_id, created_at)")
        have = {r[1] for r in conn.execute("PRAGMA table_info(docs)")}
        for col, ddl in (("owner_id", "TEXT"), ("name", "TEXT"), ("pages", "INTEGER"), ("method", "TEXT"),
                         ("tables_n", "INTEGER"), ("figures_n", "INTEGER")):
            if col not in have:
                conn.execute(f"ALTER TABLE docs ADD COLUMN {col} {ddl}")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_docs_owner ON docs(owner_id)")
        version = 3
    if version != start:
        conn.execute(f"PRAGMA user_version = {version}")
    conn.commit()
    return version


_ready: set = set()  # database files already brought up to SCHEMA_VERSION by this process
_ready_lock = threading.Lock()


def init_db(path=None) -> None:
    """Make sure the schema is current. Every storage function calls this, so after the first call per file
    it is only a set lookup and a file check (a deleted file, as in tests, is created again)."""
    p = Path(path or config.DB_PATH)
    key = str(p.resolve()) if p.exists() else None
    if key in _ready:
        return
    with _ready_lock:
        with closing(_connect(p)) as conn:
            migrate(conn)
        _ready.add(str(p.resolve()))


def delete_owner_data(owner: str, path=None) -> dict:
    """Remove everything stored for one owner (profile or session): results, saved quizzes, review cards."""
    init_db(path)
    with closing(_connect(path)) as conn:
        counts = {}
        for table, col in (("results", "session_id"), ("quizzes", "owner"), ("srs", "owner")):
            counts[table] = conn.execute(f"DELETE FROM {table} WHERE {col} = ?", (owner,)).rowcount
        conn.commit()
    return counts


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
