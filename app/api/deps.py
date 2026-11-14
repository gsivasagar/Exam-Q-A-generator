"""Request helpers: who is calling, with which model credentials, and from where."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from fastapi import Request

from .. import config
from ..auth import sessions, tokens, users
from ..auth.ratelimit import RateLimiter
from .errors import ApiProblem

limiter = RateLimiter()
COOKIE_NAME = "qf_refresh"
CSRF_HEADER = "x-requested-with"
CSRF_VALUE = "quizforge"


@dataclass(frozen=True)
class Principal:
    user: users.User
    session_id: str


@dataclass(frozen=True)
class ModelCredentials:
    provider: str
    api_key: str


def client_ip(request: Request) -> str:
    """The caller's address. Behind N trusted proxies, take the entry N from the right in X-Forwarded-For
    (anything further left was supplied by the caller and can be made up)."""
    hops = config.TRUSTED_PROXY_HOPS
    forwarded = request.headers.get("x-forwarded-for", "")
    if hops > 0 and forwarded:
        parts = [p.strip() for p in forwarded.split(",") if p.strip()]
        if len(parts) >= hops:
            return parts[-hops]
    return request.client.host if request.client else "unknown"


def unauthorized(message: str = "Sign in to continue.") -> ApiProblem:
    return ApiProblem("unauthorized", message, headers={"WWW-Authenticate": "Bearer"})


def current_principal(request: Request) -> Principal:
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise unauthorized()
    try:
        claims = tokens.decode_access_token(token.strip())
    except tokens.InvalidToken:
        raise unauthorized("Your session ended. Sign in again.") from None
    # A signed token is not enough: the session must still exist and not be revoked.
    if not sessions.is_active(claims.session_id, claims.user_id):
        raise unauthorized("Your session ended. Sign in again.")
    user = users.get_user(claims.user_id)
    if user is None or users.is_expired(user):
        raise unauthorized("Your session ended. Sign in again.")
    return Principal(user, claims.session_id)


def model_credentials(request: Request) -> ModelCredentials:
    provider = request.headers.get("x-provider", "Gemini")
    if provider not in ("Gemini", "Ollama"):
        raise ApiProblem("invalid_request", "Unknown model provider.")
    return ModelCredentials(provider, request.headers.get("x-api-key", "").strip())


def require_same_origin(request: Request) -> None:
    """For the two endpoints that act on the refresh cookie: the browser must say the call comes from our own
    page (Origin) and send a header a cross-site form cannot add."""
    origin: Optional[str] = request.headers.get("origin")
    if origin != config.APP_ORIGIN or request.headers.get(CSRF_HEADER) != CSRF_VALUE:
        raise ApiProblem("csrf", "This request did not come from the QuizForge page.")


def throttle(key: str, limit: int, window_seconds: int) -> None:
    if not limiter.allow(key, limit, window_seconds):
        wait = limiter.retry_after(key, window_seconds)
        raise ApiProblem(
            "rate_limited", f"Too many attempts. Try again in {wait} seconds.", headers={"Retry-After": str(wait)}
        )
