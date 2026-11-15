"""Login sessions. The browser holds a random refresh token in a cookie; only its hash is stored here.

Every refresh issues a new token and keeps the previous hash. Presenting an old token later means a copy
was used, so the whole session is revoked. A second refresh within a few seconds (two tabs, or React
strict mode in development) is allowed once without issuing another token.
"""

from __future__ import annotations

import uuid
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from .. import config
from ..database import _connect, init_db
from .tokens import hash_token, new_refresh_token

GRACE_SECONDS = 10


class InvalidRefresh(Exception):
    """Unknown, expired or revoked refresh token."""


class ReuseDetected(Exception):
    """An already-replaced refresh token was used again; the session has been revoked."""


@dataclass(frozen=True)
class Session:
    id: str
    user_id: str
    user_agent: str
    created_at: str
    last_used_at: str
    remember: bool = False


@dataclass(frozen=True)
class Rotation:
    session_id: str
    user_id: str
    new_token: Optional[str]  # None inside the grace window: the cookie from the first refresh is still current
    expires_at: str
    remember: bool


def _t(now: Optional[datetime]) -> datetime:
    return now or datetime.now(timezone.utc)


def create_session(user_id: str, *, remember: bool, user_agent: str = "", now: Optional[datetime] = None, path=None):
    """Returns (session_id, refresh_token, expires_at). A session without `remember` is also limited on the
    server (SESSION_TTL_HOURS), so a stolen cookie cannot outlive it."""
    t = _t(now)
    life = timedelta(days=config.REFRESH_TTL_DAYS) if remember else timedelta(hours=config.SESSION_TTL_HOURS)
    session_id, token = str(uuid.uuid4()), new_refresh_token()
    init_db(path)
    with closing(_connect(path)) as conn:
        conn.execute(
            "INSERT INTO sessions (id, user_id, token_hash, user_agent, remember, created_at, last_used_at,"
            " expires_at) VALUES (?,?,?,?,?,?,?,?)",
            (
                session_id,
                user_id,
                hash_token(token),
                user_agent[:300],
                int(remember),
                t.isoformat(),
                t.isoformat(),
                (t + life).isoformat(),
            ),
        )
        conn.commit()
    return session_id, token, (t + life).isoformat()


def rotate(refresh_token: str, *, now: Optional[datetime] = None, path=None) -> Rotation:
    t = _t(now)
    h = hash_token(refresh_token)
    init_db(path)
    with closing(_connect(path)) as conn:
        row = conn.execute(
            "SELECT id, user_id, token_hash, prev_hash, rotated_at, expires_at, revoked_at, remember FROM sessions"
            " WHERE token_hash = ? OR prev_hash = ?",
            (h, h),
        ).fetchone()
        if row is None:
            raise InvalidRefresh()
        sid, uid, current, _prev, rotated_at, expires_at, revoked_at, remember = row
        if revoked_at or datetime.fromisoformat(expires_at) <= t:
            raise InvalidRefresh()
        if h == current:
            token = new_refresh_token()
            # Replace the token only if it is still the current one (compare and swap). When two refreshes read
            # the same row at once, one wins; the other is answered like a refresh in the grace window (no new
            # token), so the browser never keeps a token the server has already replaced.
            won = conn.execute(
                "UPDATE sessions SET prev_hash = ?, token_hash = ?, rotated_at = ?, last_used_at = ?"
                " WHERE id = ? AND token_hash = ?",
                (current, hash_token(token), t.isoformat(), t.isoformat(), sid, current),
            ).rowcount
            conn.commit()
            return Rotation(sid, uid, token if won else None, expires_at, bool(remember))
        # The presented token is the previous one: a retry just after rotation, or a stolen copy.
        if rotated_at and (t - datetime.fromisoformat(rotated_at)).total_seconds() <= GRACE_SECONDS:
            return Rotation(sid, uid, None, expires_at, bool(remember))
        conn.execute("UPDATE sessions SET revoked_at = ? WHERE id = ?", (t.isoformat(), sid))
        conn.commit()
    raise ReuseDetected()


def revoke_by_token(refresh_token: str, *, now: Optional[datetime] = None, path=None) -> bool:
    """Sign out the session this refresh token belongs to (current or just-replaced token)."""
    h = hash_token(refresh_token)
    init_db(path)
    with closing(_connect(path)) as conn:
        n = conn.execute(
            "UPDATE sessions SET revoked_at = ? WHERE (token_hash = ? OR prev_hash = ?) AND revoked_at IS NULL",
            (_t(now).isoformat(), h, h),
        ).rowcount
        conn.commit()
    return n > 0


def is_active(session_id: str, user_id: str, *, now: Optional[datetime] = None, path=None) -> bool:
    init_db(path)
    with closing(_connect(path)) as conn:
        row = conn.execute(
            "SELECT expires_at, revoked_at FROM sessions WHERE id = ? AND user_id = ?", (session_id, user_id)
        ).fetchone()
    return bool(row) and not row[1] and datetime.fromisoformat(row[0]) > _t(now)


def revoke(session_id: str, *, user_id: Optional[str] = None, now: Optional[datetime] = None, path=None) -> bool:
    """Revoke one session. With `user_id` the session must belong to that user."""
    init_db(path)
    sql, args = (
        "UPDATE sessions SET revoked_at = ? WHERE id = ? AND revoked_at IS NULL",
        [_t(now).isoformat(), session_id],
    )
    if user_id:
        sql += " AND user_id = ?"
        args.append(user_id)
    with closing(_connect(path)) as conn:
        n = conn.execute(sql, args).rowcount
        conn.commit()
    return n > 0


def revoke_others(user_id: str, keep_session_id: Optional[str], *, now: Optional[datetime] = None, path=None) -> int:
    init_db(path)
    with closing(_connect(path)) as conn:
        n = conn.execute(
            "UPDATE sessions SET revoked_at = ? WHERE user_id = ? AND revoked_at IS NULL AND id != ?",
            (_t(now).isoformat(), user_id, keep_session_id or ""),
        ).rowcount
        conn.commit()
    return n


def list_sessions(user_id: str, *, now: Optional[datetime] = None, path=None) -> List[Session]:
    init_db(path)
    with closing(_connect(path)) as conn:
        rows = conn.execute(
            "SELECT id, user_id, user_agent, created_at, last_used_at FROM sessions"
            " WHERE user_id = ? AND revoked_at IS NULL AND expires_at > ? ORDER BY last_used_at DESC",
            (user_id, _t(now).isoformat()),
        ).fetchall()
    return [Session(*r) for r in rows]
