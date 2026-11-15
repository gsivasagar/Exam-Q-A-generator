"""Documents an account has uploaded. The `docs` table is the authority on who owns a document set:
the API resolves every document id through `owned()` before touching the vector store."""

from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

from .database import _connect, init_db

_COLS = "doc_id, name, pages, method, tables_n, figures_n"


class LimitReached(Exception):
    """Recording this document would take the account past its file limit."""


class NotOwned(LookupError):
    """A document id does not exist or belongs to someone else. The API answers 404 for both."""


def _to_dict(row) -> Dict:
    return {
        "id": row[0],
        "name": row[1],
        "pages": row[2],
        "method": row[3] or "text",
        "tables": row[4] or 0,
        "figures": row[5] or 0,
    }


def register(
    owner_id: str,
    doc_id: str,
    *,
    name: str,
    pages: Optional[int],
    method: str,
    tables: int = 0,
    figures: int = 0,
    limit: Optional[int] = None,
    path=None,
) -> Dict:
    """Record a document. With `limit`, counting and inserting happen in one write transaction, so uploads that
    finish at the same moment cannot together pass the limit (LimitReached)."""
    init_db(path)
    with closing(_connect(path)) as conn:
        conn.isolation_level = None  # this function manages its own transaction
        conn.execute("BEGIN IMMEDIATE")  # take the write lock before counting
        try:
            if limit is not None:
                have = conn.execute("SELECT COUNT(*) FROM docs WHERE owner_id = ?", (owner_id,)).fetchone()[0]
                if have >= limit:
                    raise LimitReached(owner_id)
            conn.execute(
                "INSERT OR REPLACE INTO docs (doc_id, created_at, owner_id, name, pages, method, tables_n, figures_n)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (doc_id, datetime.now(timezone.utc).isoformat(), owner_id, name, pages, method, tables, figures),
            )
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
    return {"id": doc_id, "name": name, "pages": pages, "method": method, "tables": tables, "figures": figures}


def list_documents(owner_id: str, path=None) -> List[Dict]:
    init_db(path)
    with closing(_connect(path)) as conn:
        rows = conn.execute(f"SELECT {_COLS} FROM docs WHERE owner_id = ? ORDER BY created_at", (owner_id,)).fetchall()
    return [_to_dict(r) for r in rows]


def count(owner_id: str, path=None) -> int:
    init_db(path)
    with closing(_connect(path)) as conn:
        return conn.execute("SELECT COUNT(*) FROM docs WHERE owner_id = ?", (owner_id,)).fetchone()[0]


def owned(owner_id: str, doc_ids: Sequence[str], path=None) -> List[Dict]:
    """The documents with these ids, in order, if the account owns every one; otherwise NotOwned."""
    ids = list(dict.fromkeys(doc_ids))
    if not ids:
        raise NotOwned("no documents")
    init_db(path)
    with closing(_connect(path)) as conn:
        marks = ",".join("?" * len(ids))
        rows = conn.execute(
            f"SELECT {_COLS} FROM docs WHERE owner_id = ? AND doc_id IN ({marks})", [owner_id, *ids]
        ).fetchall()
    found = {r[0]: _to_dict(r) for r in rows}
    if len(found) != len(ids):
        raise NotOwned("document not found")
    return [found[i] for i in ids]


def delete(owner_id: str, doc_id: str, path=None) -> bool:
    init_db(path)
    with closing(_connect(path)) as conn:
        n = conn.execute("DELETE FROM docs WHERE doc_id = ? AND owner_id = ?", (doc_id, owner_id)).rowcount
        conn.commit()
    if n:
        from .vector_store import delete_doc

        delete_doc(doc_id)
    return n > 0
