"""Spaced repetition: resurface questions a student missed (and ones they got right, less often).

A simplified SM-2 scheduler. Each question is a card; after grading its next due date moves out
(pass) or resets to tomorrow (fail)."""
from __future__ import annotations

import hashlib
import json
from contextlib import closing
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from . import config
from .database import _connect, init_db

START_EASE, MIN_EASE, MAX_EASE = 2.5, 1.3, 2.8


def card_id(q: Dict) -> str:
    return hashlib.sha1(f"{q.get('question', '')}\x00{q.get('answer', '')}".encode("utf-8")).hexdigest()


def next_state(reps: int, interval: float, ease: float, score: float) -> Dict:
    """New (reps, interval_days, ease) after answering with `score` (0-1)."""
    if score < config.PASS_THRESHOLD:
        return {"reps": 0, "interval": 1.0, "ease": max(MIN_EASE, ease - 0.2)}
    reps += 1
    interval = 1.0 if reps == 1 else 3.0 if reps == 2 else round(max(interval, 1.0) * ease)
    return {"reps": reps, "interval": float(interval), "ease": min(MAX_EASE, ease + 0.05)}


def record_results(owner: str, graded: List[Dict], *, now: Optional[datetime] = None, path=None) -> int:
    """Update cards from graded answers (ungraded ones are skipped). Returns how many cards changed."""
    init_db(path)
    now = now or datetime.now(timezone.utc)
    changed = 0
    with closing(_connect(path)) as conn:
        for g in graded:
            if g.get("score") is None:
                continue
            cid = card_id(g)
            row = conn.execute("SELECT reps, interval_days, ease FROM srs WHERE owner = ? AND qhash = ?",
                               (owner, cid)).fetchone()
            reps, interval, ease = row if row else (0, 0.0, START_EASE)
            st = next_state(reps, interval, ease, g["score"])
            card = {k: v for k, v in g.items() if k not in ("student", "score", "feedback", "error", "covered")}
            conn.execute(
                "INSERT OR REPLACE INTO srs (owner, qhash, question_json, due_at, interval_days, ease, reps,"
                " last_score) VALUES (?,?,?,?,?,?,?,?)",
                (owner, cid, json.dumps(card), (now + timedelta(days=st["interval"])).isoformat(),
                 st["interval"], st["ease"], st["reps"], g["score"]))
            changed += 1
        conn.commit()
    return changed


def due_questions(owner: str, *, now: Optional[datetime] = None, limit: int = 10, path=None) -> List[Dict]:
    """Questions whose review date has arrived, most overdue first."""
    init_db(path)
    now = now or datetime.now(timezone.utc)
    with closing(_connect(path)) as conn:
        rows = conn.execute("SELECT question_json FROM srs WHERE owner = ? AND due_at <= ? ORDER BY due_at LIMIT ?",
                            (owner, now.isoformat(), limit)).fetchall()
    return [json.loads(r[0]) for r in rows]


def count_due(owner: str, *, now: Optional[datetime] = None, path=None) -> int:
    init_db(path)
    now = now or datetime.now(timezone.utc)
    with closing(_connect(path)) as conn:
        return conn.execute("SELECT COUNT(*) FROM srs WHERE owner = ? AND due_at <= ?",
                            (owner, now.isoformat())).fetchone()[0]
