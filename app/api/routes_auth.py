"""Sign up, sign in, guests, refresh and sign out."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Request, Response
from pydantic import BaseModel, Field

from .. import config
from ..auth import passwords, sessions, tokens, users
from .deps import COOKIE_NAME, Principal, client_ip, current_principal, require_same_origin, throttle, unauthorized
from .errors import ApiProblem

router = APIRouter(prefix="/auth", tags=["auth"])


class Credentials(BaseModel):
    email: str = Field(max_length=320)
    password: str = Field(max_length=1024)  # the policy gives the real limit; this only bounds the request size
    remember: bool = True


class Upgrade(BaseModel):
    email: str = Field(max_length=320)
    password: str = Field(max_length=1024)


def user_json(user: users.User) -> dict:
    return {"id": user.id, "email": user.email, "isGuest": user.is_guest, "expiresAt": user.expires_at}


def _set_cookie(response: Response, token: str, *, remember: bool) -> None:
    response.set_cookie(
        COOKIE_NAME,
        token,
        httponly=True,
        secure=config.COOKIE_SECURE,
        samesite="lax",
        path="/api/auth",
        max_age=config.REFRESH_TTL_DAYS * 86400 if remember else None,
    )


def _clear_cookie(response: Response) -> None:
    response.delete_cookie(COOKIE_NAME, path="/api/auth", httponly=True, secure=config.COOKIE_SECURE, samesite="lax")


def _signed_in(response: Response, user: users.User, session_id: str, token: Optional[str], remember: bool) -> dict:
    if token:
        _set_cookie(response, token, remember=remember)
    return {
        "accessToken": tokens.issue_access_token(user.id, session_id),
        "expiresIn": config.ACCESS_TTL_MIN * 60,
        "user": user_json(user),
    }


def _start(request: Request, response: Response, user: users.User, remember: bool) -> dict:
    sid, token, _ = sessions.create_session(
        user.id, remember=remember, user_agent=request.headers.get("user-agent", "")
    )
    return _signed_in(response, user, sid, token, remember)


@router.post("/signup")
def signup(body: Credentials, request: Request, response: Response):
    throttle(f"signup:{client_ip(request)}", 5, 3600)
    try:
        users.normalize_email(body.email)
        passwords.check_policy(body.password, body.email)
    except users.InvalidEmail as exc:
        raise ApiProblem("invalid_email", str(exc)) from None
    except passwords.PasswordPolicyError as exc:
        raise ApiProblem("weak_password", str(exc)) from None
    try:
        user = users.create_user(body.email, passwords.hash_password(body.password))
    except users.EmailTaken:
        raise ApiProblem("email_taken", "An account with this email already exists. Try signing in.") from None
    return _start(request, response, user, body.remember)


@router.post("/login")
def login(body: Credentials, request: Request, response: Response):
    ip = client_ip(request)
    throttle(f"login-ip:{ip}", 20, 3600)
    throttle(f"login:{ip}:{body.email.strip().lower()}", 5, 60)
    found = users.get_with_hash(body.email)
    if found is None:
        passwords.burn_time()  # a missing account must not answer faster than a wrong password
        raise ApiProblem("invalid_credentials", "Email or password is wrong. Check both and try again.")
    user, stored = found
    if not passwords.verify_password(body.password, stored):
        raise ApiProblem("invalid_credentials", "Email or password is wrong. Check both and try again.")
    if passwords.needs_rehash(stored):
        users.set_password(user.id, passwords.hash_password(body.password))
    return _start(request, response, user, body.remember)


@router.post("/guest")
def guest(request: Request, response: Response):
    throttle(f"guest:{client_ip(request)}", 5, 3600)
    return _start(request, response, users.create_guest(), remember=True)


@router.post("/upgrade")
def upgrade(body: Upgrade, response: Response, who: Principal = Depends(current_principal)):
    """A guest gives an email and password. Same user id, so their files and quizzes stay."""
    if not who.user.is_guest:
        raise ApiProblem("not_guest", "This account already has an email and password.")
    try:
        users.normalize_email(body.email)
        passwords.check_policy(body.password, body.email)
    except users.InvalidEmail as exc:
        raise ApiProblem("invalid_email", str(exc)) from None
    except passwords.PasswordPolicyError as exc:
        raise ApiProblem("weak_password", str(exc)) from None
    try:
        user = users.upgrade_guest(who.user.id, body.email, passwords.hash_password(body.password))
    except users.EmailTaken:
        raise ApiProblem("email_taken", "An account with this email already exists. Try signing in.") from None
    return _signed_in(response, user, who.session_id, None, True)


@router.post("/refresh")
def refresh(request: Request, response: Response):
    require_same_origin(request)
    cookie = request.cookies.get(COOKIE_NAME)
    if not cookie:
        raise unauthorized("Sign in to continue.")
    try:
        rotation = sessions.rotate(cookie)
    except sessions.ReuseDetected:
        raise ApiProblem(
            "unauthorized", "Your session ended. Sign in again.", headers=_cookie_clearing(response)
        ) from None
    except sessions.InvalidRefresh:
        raise ApiProblem(
            "unauthorized", "Your session ended. Sign in again.", headers=_cookie_clearing(response)
        ) from None
    user = users.get_user(rotation.user_id)
    if user is None or users.is_expired(user):
        raise ApiProblem("unauthorized", "Your session ended. Sign in again.", headers=_cookie_clearing(response))
    return _signed_in(response, user, rotation.session_id, rotation.new_token, rotation.remember)


def _cookie_clearing(response: Response) -> dict:
    """Headers that delete the cookie, for error responses (FastAPI drops a Response's headers on errors)."""
    secure = "; Secure" if config.COOKIE_SECURE else ""
    return {"Set-Cookie": f"{COOKIE_NAME}=; Max-Age=0; Path=/api/auth; HttpOnly; SameSite=Lax{secure}"}


@router.post("/logout", status_code=204)
def logout(request: Request, response: Response):
    require_same_origin(request)
    cookie = request.cookies.get(COOKIE_NAME)
    if cookie:
        sessions.revoke_by_token(cookie)
    _clear_cookie(response)
    response.status_code = 204
    return response
