"""Access tokens (short-lived JWTs) and refresh tokens (random, stored only as a hash)."""

from __future__ import annotations

import hashlib
import secrets
import time
from dataclasses import dataclass
from typing import Optional

import jwt

from .. import config

ALGORITHM = "HS256"
ISSUER = "quizforge"
AUDIENCE = "quizforge-api"

_throwaway_secret: Optional[str] = None


class AuthConfigError(RuntimeError):
    pass


def secret() -> str:
    """The signing secret. Production must set AUTH_SECRET; development falls back to a random one that
    changes on every start (so old tokens stop working, never silently weak)."""
    global _throwaway_secret
    if config.AUTH_SECRET:
        if len(config.AUTH_SECRET) < 32:
            raise AuthConfigError("AUTH_SECRET must be at least 32 characters.")
        return config.AUTH_SECRET
    if config.APP_ENV == "production":
        raise AuthConfigError("AUTH_SECRET is required when APP_ENV=production.")
    if _throwaway_secret is None:
        _throwaway_secret = secrets.token_urlsafe(48)
    return _throwaway_secret


@dataclass(frozen=True)
class AccessClaims:
    user_id: str
    session_id: str
    expires_at: int


class InvalidToken(Exception):
    """The access token is missing, expired, forged or for another service."""


def issue_access_token(user_id: str, session_id: str, *, now: Optional[float] = None) -> str:
    t = int(now if now is not None else time.time())
    claims = {
        "sub": user_id,
        "sid": session_id,
        "iat": t,
        "exp": t + config.ACCESS_TTL_MIN * 60,
        "iss": ISSUER,
        "aud": AUDIENCE,
    }
    return jwt.encode(claims, secret(), algorithm=ALGORITHM)


def decode_access_token(token: str) -> AccessClaims:
    try:
        # The algorithm is pinned: a token that says "alg: none" or names another algorithm is refused.
        data = jwt.decode(
            token,
            secret(),
            algorithms=[ALGORITHM],
            audience=AUDIENCE,
            issuer=ISSUER,
            options={"require": ["exp", "iat", "sub", "sid", "iss", "aud"]},
        )
    except jwt.PyJWTError as exc:
        raise InvalidToken(str(exc)) from exc
    if not isinstance(data.get("sub"), str) or not isinstance(data.get("sid"), str):
        raise InvalidToken("malformed claims")
    return AccessClaims(data["sub"], data["sid"], int(data["exp"]))


def new_refresh_token() -> str:
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
