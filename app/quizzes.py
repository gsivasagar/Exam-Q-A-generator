"""Saved quizzes: store generated questions so a quiz can be retaken or reviewed later."""
from __future__ import annotations

import json
import uuid
from contextlib import closing
from datetime import datetime, timezone
from typing import Dict, List, Optional

from .database import _connect, init_db


def save_quiz(owner: str, questions: List[Dict], *, title: str = "", question_type: str = "",
              difficulty: Optional[str] = None, quiz_id: Optional[str] = None, path=None) -> str:
    """Store a quiz; returns its id. Saving the same id again replaces it."""
    init_db(path)
    quiz_id = quiz_id or str(uuid.uuid4())
    with closing(_connect(path)) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO quizzes (id, owner, created_at, title, question_type, difficulty, n,"
            " questions_json) VALUES (?,?,?,?,?,?,?,?)",
            (quiz_id, owner, datetime.now(timezone.utc).isoformat(), title or "Untitled quiz", question_type,
             difficulty, len(questions), json.dumps(questions)))
        conn.commit()
    return quiz_id


def list_quizzes(owner: str, limit: int = 50, path=None) -> List[Dict]:
    """Newest first, each with its attempt count and average score from the results table."""
    init_db(path)
    with closing(_connect(path)) as conn:
        rows = conn.execute(
            "SELECT q.id, q.title, q.created_at, q.question_type, q.difficulty, q.n,"
            " (SELECT COUNT(DISTINCT timestamp) FROM results r WHERE r.quiz_id = q.id AND r.session_id = q.owner),"
            " (SELECT AVG(score) FROM results r WHERE r.quiz_id = q.id AND r.session_id = q.owner)"
            " FROM quizzes q WHERE q.owner = ? ORDER BY q.created_at DESC LIMIT ?", (owner, limit)).fetchall()
    keys = ("id", "title", "created_at", "question_type", "difficulty", "n", "attempts", "avg_score")
    return [dict(zip(keys, r)) for r in rows]


def load_quiz(owner: str, quiz_id: str, path=None) -> Optional[Dict]:
    """The saved quiz {id, title, questions, ...} if it belongs to `owner`, else None."""
    init_db(path)
    with closing(_connect(path)) as conn:
        row = conn.execute("SELECT id, title, question_type, difficulty, questions_json FROM quizzes"
                           " WHERE id = ? AND owner = ?", (quiz_id, owner)).fetchone()
    if row is None:
        return None
    return {"id": row[0], "title": row[1], "question_type": row[2], "difficulty": row[3],
            "questions": json.loads(row[4])}


def delete_quiz(owner: str, quiz_id: str, path=None) -> bool:
    init_db(path)
    with closing(_connect(path)) as conn:
        n = conn.execute("DELETE FROM quizzes WHERE id = ? AND owner = ?", (quiz_id, owner)).rowcount
        conn.commit()
    return n > 0
