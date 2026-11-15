"""Make a quiz from your files, take it, grade it, see the result."""

from __future__ import annotations

import json
import uuid
from contextlib import closing
from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from .. import documents
from ..concurrency import effective_workers
from ..database import _connect, init_db, store_results
from ..grader import grade_all
from ..llm import LLMError
from ..qa_generator import generate_qa_pairs
from ..quizzes import list_quizzes, load_quiz, save_quiz
from . import llm_factory
from .deps import ModelCredentials, Principal, current_principal, model_credentials
from .errors import ApiProblem, model_problem

router = APIRouter(prefix="/quizzes", tags=["quizzes"])


class CreateQuiz(BaseModel):
    documentIds: List[str] = Field(min_length=1, max_length=20)
    count: int = Field(ge=5, le=25)
    type: Literal["short", "mcq", "mixed", "tf", "cloze"] = "mcq"
    difficulty: Literal["any", "easy", "medium", "hard"] = "any"
    topic: str = Field(default="", max_length=200)


class Grade(BaseModel):
    answers: Dict[str, str] = Field(max_length=100)


def _question(i: int, qa: dict) -> dict:
    out = {
        "id": f"q{i}",
        "type": qa.get("type", "short"),
        "question": qa["question"],
        "topic": qa.get("topic") or "General",
        "source": {"file": qa.get("source") or "your files", "page": int(qa.get("page") or 0)},
    }
    if qa.get("options"):  # absent, not null, for questions without choices (the page's type says optional)
        out["options"] = qa["options"]
    return out


def _quiz_json(quiz_id: str, title: str, created_at: str, qas: List[dict]) -> dict:
    return {
        "id": quiz_id,
        "title": title,
        "createdAt": created_at,
        "questions": [_question(i, q) for i, q in enumerate(qas)],
    }


def _created_at(owner_id: str, quiz_id: str) -> str:
    init_db()
    with closing(_connect()) as conn:
        row = conn.execute("SELECT created_at FROM quizzes WHERE id = ? AND owner = ?", (quiz_id, owner_id)).fetchone()
    return row[0] if row else ""


def _own_quiz(who: Principal, quiz_id: str) -> dict:
    quiz = load_quiz(who.user.id, quiz_id)
    if quiz is None:  # missing and someone else's look the same
        raise ApiProblem("not_found", "That quiz was not found.")
    return quiz


@router.post("")
def create(
    body: CreateQuiz, who: Principal = Depends(current_principal), creds: ModelCredentials = Depends(model_credentials)
):
    try:
        docs = documents.owned(who.user.id, body.documentIds)
    except documents.NotOwned:
        raise ApiProblem("not_found", "One of those files was not found.") from None
    llm = llm_factory.make_llm(creds)
    try:
        qas = generate_qa_pairs(
            [d["id"] for d in docs],  # type: ignore[arg-type]  # a list of ids searches several files at once
            llm,
            n=body.count,
            topic=body.topic.strip() or None,
            question_type=body.type,
            difficulty=None if body.difficulty == "any" else body.difficulty,
        )
    except LLMError as exc:  # bad key, quota used up, or the provider is down
        raise model_problem(exc) from None
    except ValueError as exc:  # no usable questions, or nothing matched the topic
        raise ApiProblem("invalid_request", str(exc)) from None
    title = ("Practice: " + ", ".join(d["name"] for d in docs))[:80]
    quiz_id = save_quiz(
        who.user.id,
        qas,
        title=title,
        question_type=body.type,
        difficulty=None if body.difficulty == "any" else body.difficulty,
    )
    return _quiz_json(quiz_id, title, _created_at(who.user.id, quiz_id), qas)


def _last_scores(owner_id: str) -> Dict[str, float]:
    init_db()
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT quiz_id, result_json FROM attempts WHERE owner_id = ? ORDER BY created_at", (owner_id,)
        ).fetchall()
    return {quiz_id: json.loads(blob)["average"] for quiz_id, blob in rows}  # later attempts overwrite earlier ones


@router.get("")
def mine(who: Principal = Depends(current_principal)):
    scores = _last_scores(who.user.id)
    return [
        {
            "id": q["id"],
            "title": q["title"],
            "createdAt": q["created_at"],
            "questionCount": q["n"],
            "lastScore": scores.get(q["id"]),
        }
        for q in list_quizzes(who.user.id)
    ]


@router.get("/{quiz_id}")
def one(quiz_id: str, who: Principal = Depends(current_principal)):
    quiz = _own_quiz(who, quiz_id)
    return _quiz_json(quiz["id"], quiz["title"], _created_at(who.user.id, quiz_id), quiz["questions"])


def _status(score: Optional[float]) -> str:
    if score is None:
        return "ungraded"
    return "correct" if score >= 0.999 else "partial" if score > 0 else "incorrect"


@router.post("/{quiz_id}/grade")
def grade(
    quiz_id: str,
    body: Grade,
    who: Principal = Depends(current_principal),
    creds: ModelCredentials = Depends(model_credentials),
):
    quiz = _own_quiz(who, quiz_id)
    qas: List[dict] = quiz["questions"]
    answers = [body.answers.get(f"q{i}", "")[:5000] for i in range(len(qas))]
    needs_model = any(q.get("type") not in ("mcq", "tf") and a.strip() for q, a in zip(qas, answers))
    llm = llm_factory.make_llm(creds) if needs_model else None
    graded = grade_all(qas, answers, llm, workers=effective_workers(llm))  # type: ignore[arg-type]

    results: List[dict] = []
    by_topic: Dict[str, List[float]] = {}
    for i, (qa, g, given) in enumerate(zip(qas, graded, answers)):
        score = g.get("score")
        points = qa.get("key_points") or []
        covered = g.get("covered")
        item = {
            "questionId": f"q{i}",
            "question": qa["question"],
            "source": _question(i, qa)["source"],
            "status": _status(score),
            "score": score if score is not None else 0.0,
            "yourAnswer": given,
        }
        if qa.get("answer"):
            item["correctAnswer"] = qa["answer"]
        if covered and len(covered) == len(points):
            item["keyPoints"] = [{"text": p, "covered": bool(c)} for p, c in zip(points, covered)]
        elif qa.get("type") not in ("mcq", "tf") and g.get("feedback"):
            item["explanation"] = g["feedback"]
        if score is None:
            item["explanation"] = g.get("feedback", "Could not grade this answer automatically.")
        results.append(item)
        by_topic.setdefault(qa.get("topic") or "General", []).append(item["score"])

    averages = {t: sum(v) / len(v) for t, v in by_topic.items()}  # weakest topic first once sorted
    result = {
        "quizId": quiz_id,
        "title": quiz["title"],
        "average": sum(r["score"] for r in results) / max(len(results), 1),
        "correct": sum(1 for r in results if r["status"] == "correct"),
        "total": len(results),
        "results": results,
        "topics": [{"topic": t, "score": s} for t, s in sorted(averages.items(), key=lambda kv: kv[1])],
    }
    store_results(
        [
            {**g, "question": qa["question"], "topic": qa.get("topic"), "answer": a}
            for qa, g, a in zip(qas, graded, answers)
        ],
        who.user.id,
        quiz_id,
    )
    init_db()
    with closing(_connect()) as conn:
        conn.execute(
            "INSERT INTO attempts (id, owner_id, quiz_id, created_at, result_json) VALUES (?,?,?,?,?)",
            (str(uuid.uuid4()), who.user.id, quiz_id, datetime.now(timezone.utc).isoformat(), json.dumps(result)),
        )
        conn.commit()
    return result


@router.get("/{quiz_id}/result")
def latest_result(quiz_id: str, who: Principal = Depends(current_principal)):
    _own_quiz(who, quiz_id)
    init_db()
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT result_json FROM attempts WHERE owner_id = ? AND quiz_id = ? ORDER BY created_at DESC LIMIT 1",
            (who.user.id, quiz_id),
        ).fetchone()
    if row is None:
        raise ApiProblem("not_found", "There are no results for this quiz yet.")
    return json.loads(row[0])
