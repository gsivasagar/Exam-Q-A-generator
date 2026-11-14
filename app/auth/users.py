"""Users: create, look up, upgrade a guest, delete an account and everything it owns."""

from __future__ import annotations

import re
import sqlite3
import uuid
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

from .. import config
from ..database import _connect, init_db

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class EmailTaken(Exception):
    pass


class InvalidEmail(ValueError):
    pass


@dataclass(frozen=True)
class User:
    id: str
    email: Optional[str]
    is_guest: bool
    created_at: str
    expires_at: Optional[str]


def normalize_email(email: str) -> str:
    cleaned = (email or "").strip().lower()
    if len(cleaned) > 254 or not _EMAIL.match(cleaned):
        raise InvalidEmail("That does not look like an email address.")
    return cleaned


def _now(now: Optional[datetime]) -> datetime:
    return now or datetime.now(timezone.utc)


def _row_to_user(row) -> User:
    return User(id=row[0], email=row[1], is_guest=bool(row[2]), created_at=row[3], expires_at=row[4])


_COLUMNS = "id, email, is_guest, created_at, expires_at"


def create_user(email: str, password_hash: str, *, now: Optional[datetime] = None, path=None) -> User:
    email = normalize_email(email)
    user = User(str(uuid.uuid4()), email, False, _now(now).isoformat(), None)
    init_db(path)
    with closing(_connect(path)) as conn:
        try:
            conn.execute(
                "INSERT INTO users (id, email, password_hash, is_guest, created_at) VALUES (?,?,?,0,?)",
                (user.id, email, password_hash, user.created_at),
            )
            conn.commit()
        except sqlite3.IntegrityError as exc:
            raise EmailTaken(email) from exc
    return user


def create_guest(*, now: Optional[datetime] = None, path=None) -> User:
    t = _now(now)
    user = User(str(uuid.uuid4()), None, True, t.isoformat(), (t + timedelta(days=config.GUEST_TTL_DAYS)).isoformat())
    init_db(path)
    with closing(_connect(path)) as conn:
        conn.execute(
            "INSERT INTO users (id, email, password_hash, is_guest, created_at, expires_at) VALUES (?,NULL,NULL,1,?,?)",
            (user.id, user.created_at, user.expires_at),
        )
        conn.commit()
    return user


def get_user(user_id: str, path=None) -> Optional[User]:
    init_db(path)
    with closing(_connect(path)) as conn:
        row = conn.execute(f"SELECT {_COLUMNS} FROM users WHERE id = ?", (user_id,)).fetchone()
    return _row_to_user(row) if row else None


def get_with_hash(email: str, path=None) -> Optional[Tuple[User, str]]:
    """The user and password hash for a login attempt, or None. A malformed email is simply not found."""
    try:
        email = normalize_email(email)
    except InvalidEmail:
        return None
    init_db(path)
    with closing(_connect(path)) as conn:
        row = conn.execute(f"SELECT {_COLUMNS}, password_hash FROM users WHERE email = ?", (email,)).fetchone()
    return (_row_to_user(row[:5]), row[5]) if row and row[5] else None


def password_hash_of(user_id: str, path=None) -> Optional[str]:
    init_db(path)
    with closing(_connect(path)) as conn:
        row = conn.execute("SELECT password_hash FROM users WHERE id = ?", (user_id,)).fetchone()
    return row[0] if row else None


def set_password(user_id: str, password_hash: str, path=None) -> None:
    init_db(path)
    with closing(_connect(path)) as conn:
        conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (password_hash, user_id))
        conn.commit()


def upgrade_guest(user_id: str, email: str, password_hash: str, path=None) -> User:
    """Give a guest an email and password. The user id stays the same, so their files and quizzes stay."""
    email = normalize_email(email)
    init_db(path)
    with closing(_connect(path)) as conn:
        try:
            n = conn.execute(
                "UPDATE users SET email = ?, password_hash = ?, is_guest = 0, expires_at = NULL"
                " WHERE id = ? AND is_guest = 1",
                (email, password_hash, user_id),
            ).rowcount
            conn.commit()
        except sqlite3.IntegrityError as exc:
            raise EmailTaken(email) from exc
    if n == 0:
        raise ValueError("only a guest can be upgraded")
    user = get_user(user_id, path)
    assert user is not None
    return user


def is_expired(user: User, *, now: Optional[datetime] = None) -> bool:
    return bool(user.expires_at) and datetime.fromisoformat(user.expires_at) <= _now(now)  # type: ignore[arg-type]


def expired_guest_ids(*, now: Optional[datetime] = None, path=None) -> List[str]:
    init_db(path)
    with closing(_connect(path)) as conn:
        rows = conn.execute(
            "SELECT id FROM users WHERE is_guest = 1 AND expires_at <= ?", (_now(now).isoformat(),)
        ).fetchall()
    return [r[0] for r in rows]


def delete_account(user_id: str, path=None) -> dict:
    """Remove the user and everything they own: login sessions, files and their search chunks, quizzes,
    results, review cards and attempts. Returns how many rows went from each table."""
    init_db(path)
    counts = {}
    with closing(_connect(path)) as conn:
        for table, col in (
            ("sessions", "user_id"),
            ("attempts", "owner_id"),
            ("results", "session_id"),
            ("quizzes", "owner"),
            ("srs", "owner"),
            ("docs", "owner_id"),
            ("users", "id"),
        ):
            counts[table] = conn.execute(f"DELETE FROM {table} WHERE {col} = ?", (user_id,)).rowcount
        conn.commit()
    try:
        from ..vector_store import delete_owner

        delete_owner(user_id)
    except Exception:  # the vector store may never have been opened for this account
        counts["chunks_error"] = 1
    return counts


def delete_expired_guests(*, now: Optional[datetime] = None, path=None) -> int:
    ids = expired_guest_ids(now=now, path=path)
    for user_id in ids:
        delete_account(user_id, path)
    return len(ids)
