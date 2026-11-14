"""The signed-in person: their account, devices, data export and deletion."""

from __future__ import annotations

import json
from contextlib import closing

from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel, Field

from .. import documents
from ..auth import passwords, sessions, users
from ..database import _connect, init_db
from ..quizzes import list_quizzes, load_quiz
from .deps import Principal, current_principal
from .errors import ApiProblem
from .routes_auth import user_json

router = APIRouter(prefix="/me", tags=["account"])


class ChangePassword(BaseModel):
    current: str = Field(max_length=1024)
    new: str = Field(max_length=1024)


class ConfirmDelete(BaseModel):
    email: str = Field(max_length=320)


def _review_cards(owner_id: str) -> int:
    init_db()
    with closing(_connect()) as conn:
        return conn.execute("SELECT COUNT(*) FROM srs WHERE owner = ?", (owner_id,)).fetchone()[0]


@router.get("")
def me(who: Principal = Depends(current_principal)):
    return {
        **user_json(who.user),
        "counts": {
            "files": documents.count(who.user.id),
            "quizzes": len(list_quizzes(who.user.id, limit=1000)),
            "reviewCards": _review_cards(who.user.id),
        },
    }


@router.post("/password", status_code=204)
def change_password(body: ChangePassword, who: Principal = Depends(current_principal)):
    stored = users.password_hash_of(who.user.id)
    if not stored:
        raise ApiProblem("no_password", "Guest sessions have no password. Create an account first.")
    if not passwords.verify_password(body.current, stored):
        raise ApiProblem("wrong_password", "The current password is wrong.")
    try:
        passwords.check_policy(body.new, who.user.email)
    except passwords.PasswordPolicyError as exc:
        raise ApiProblem("weak_password", str(exc)) from None
    users.set_password(who.user.id, passwords.hash_password(body.new))
    sessions.revoke_others(who.user.id, who.session_id)
    return Response(status_code=204)


@router.get("/sessions")
def my_sessions(who: Principal = Depends(current_principal)):
    return [
        {
            "id": s.id,
            "userAgent": s.user_agent,
            "createdAt": s.created_at,
            "lastUsedAt": s.last_used_at,
            "current": s.id == who.session_id,
        }
        for s in sessions.list_sessions(who.user.id)
    ]


@router.delete("/sessions/others")
def sign_out_others(who: Principal = Depends(current_principal)):
    return {"signedOut": sessions.revoke_others(who.user.id, who.session_id)}


@router.delete("/sessions/{session_id}", status_code=204)
def sign_out_one(session_id: str, who: Principal = Depends(current_principal)):
    if not sessions.revoke(session_id, user_id=who.user.id):
        raise ApiProblem("not_found", "That session was not found.")
    return Response(status_code=204)


@router.get("/export")
def export(who: Principal = Depends(current_principal)):
    """Everything stored about the account, as one JSON download."""
    init_db()
    quizzes = [load_quiz(who.user.id, q["id"]) for q in list_quizzes(who.user.id, limit=1000)]
    with closing(_connect()) as conn:
        attempts = [
            {"quizId": r[0], "createdAt": r[1], "result": json.loads(r[2])}
            for r in conn.execute(
                "SELECT quiz_id, created_at, result_json FROM attempts WHERE owner_id = ? ORDER BY created_at",
                (who.user.id,),
            )
        ]
    body = {
        "account": user_json(who.user),
        "files": documents.list_documents(who.user.id),
        "quizzes": quizzes,
        "attempts": attempts,
    }
    return Response(
        json.dumps(body, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": 'attachment; filename="quizforge-data.json"'},
    )


@router.delete("", status_code=204)
def delete_me(body: ConfirmDelete, response: Response, who: Principal = Depends(current_principal)):
    if who.user.email and body.email.strip().lower() != who.user.email:
        raise ApiProblem("confirmation_mismatch", "That is not the email on this account.")
    users.delete_account(who.user.id)
    return Response(status_code=204)
