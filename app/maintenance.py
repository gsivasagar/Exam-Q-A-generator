"""Housekeeping: forget uploaded documents (vector chunks) that were never cleaned up."""
from __future__ import annotations

import logging
from contextlib import closing
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from . import config
from .database import _connect, init_db

log = logging.getLogger(__name__)


def register_doc(doc_id: str, *, now: Optional[datetime] = None, path=None) -> None:
    """Remember when a document set was created so it can expire later."""
    init_db(path)
    with closing(_connect(path)) as conn:
        conn.execute("INSERT OR REPLACE INTO docs (doc_id, created_at) VALUES (?, ?)",
                     (doc_id, (now or datetime.now(timezone.utc)).isoformat()))
        conn.commit()


def forget_doc(doc_id: str, path=None) -> None:
    init_db(path)
    with closing(_connect(path)) as conn:
        conn.execute("DELETE FROM docs WHERE doc_id = ?", (doc_id,))
        conn.commit()


def expired_docs(max_age_days: Optional[float] = None, *, now: Optional[datetime] = None, path=None) -> List[str]:
    days = config.DOC_TTL_DAYS if max_age_days is None else max_age_days
    cutoff = ((now or datetime.now(timezone.utc)) - timedelta(days=days)).isoformat()
    init_db(path)
    with closing(_connect(path)) as conn:
        return [r[0] for r in conn.execute("SELECT doc_id FROM docs WHERE created_at < ?", (cutoff,))]


def cleanup_expired(max_age_days: Optional[float] = None, *, now: Optional[datetime] = None, path=None) -> int:
    """Delete vector-store chunks of documents older than the TTL. Returns how many were removed.
    A TTL of 0 or less disables cleanup."""
    days = config.DOC_TTL_DAYS if max_age_days is None else max_age_days
    if days <= 0:
        return 0
    from .vector_store import delete_doc

    removed = 0
    for doc_id in expired_docs(days, now=now, path=path):
        try:
            delete_doc(doc_id)
        except Exception as exc:  # a broken store must not stop the app from starting
            log.warning("Could not delete expired document %s: %s", doc_id, exc)
            continue
        forget_doc(doc_id, path=path)
        removed += 1
    if removed:
        log.info("Removed %d expired document set(s)", removed)
    return removed
