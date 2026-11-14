"""The front end's TypeScript types (frontend/src/api/types.ts) against what the server really sends.

The types file is parsed, so a field renamed on one side and not the other fails here instead of in a browser."""

import json
import re
from pathlib import Path

import pytest

from tests.test_api import client, join, make_doc, make_quiz  # noqa: F401

TYPES = Path(__file__).resolve().parent.parent / "frontend" / "src" / "api" / "types.ts"


def parse_interfaces(text: str) -> dict:
    """{name: {field: (optional, nullable)}} for every `export interface`, with `extends` resolved."""
    found, parents = {}, {}
    for m in re.finditer(r"export interface (\w+)(?: extends (\w+))? \{\n(.*?)\n\}", text, re.S):
        name, parent, body = m.groups()
        fields = {}
        for line in body.splitlines():
            f = re.match(r"\s{2}(\w+)(\?)?: (.+?)\s*$", line)
            if f and not line.strip().startswith(("//", "/*", "*")):
                fields[f.group(1)] = (bool(f.group(2)), "null" in f.group(3))
        found[name], parents[name] = fields, parent
    for name, parent in parents.items():
        if parent:
            found[name] = {**found[parent], **found[name]}
    return found


@pytest.fixture(scope="module")
def types():
    if not TYPES.exists():
        pytest.skip("front end not present")
    return parse_interfaces(TYPES.read_text())


def check(types, name, value, where=""):
    spec = types[name]
    assert isinstance(value, dict), f"{where or name} is not an object"
    required = {k for k, (opt, _) in spec.items() if not opt}
    missing = required - set(value)
    extra = set(value) - set(spec)
    assert not missing, f"{where or name}: missing {sorted(missing)}"
    assert not extra, f"{where or name}: not in the TypeScript type {sorted(extra)}"
    for key, val in value.items():
        optional, nullable = spec[key]
        if val is None:
            assert nullable, f"{where or name}.{key} is null but the TypeScript type does not allow null"


def test_parser_reads_the_types(types):
    assert {
        "AuthResult",
        "User",
        "Me",
        "Device",
        "IngestedDocument",
        "Quiz",
        "Question",
        "QuizSummary",
        "GradeResult",
        "QuestionResult",
        "TopicScore",
        "SourceRef",
    } <= set(types)
    assert types["Me"]["email"] == (False, True)  # inherited from User, and nullable
    assert types["Question"]["options"][0] is True  # optional


def test_every_response_matches_its_type(client, types):  # noqa: F811
    r = client.post("/api/auth/signup", json={"email": "c@example.com", "password": "a long test passphrase"})
    body = r.json()
    check(types, "AuthResult", body)
    check(types, "User", body["user"], "AuthResult.user")
    headers = {"Authorization": f"Bearer {body['accessToken']}"}

    guest = client.post("/api/auth/guest").json()
    check(types, "User", guest["user"], "guest user")

    me = client.get("/api/me", headers=headers).json()
    check(types, "Me", me)
    assert set(me["counts"]) == {"files", "quizzes", "reviewCards"}

    for device in client.get("/api/me/sessions", headers=headers).json():
        check(types, "Device", device)

    doc = make_doc(client, headers)
    check(types, "IngestedDocument", doc)
    for d in client.get("/api/documents", headers=headers).json():
        check(types, "IngestedDocument", d)

    quiz = make_quiz(client, headers, [doc["id"]])
    check(types, "Quiz", quiz)
    for q in quiz["questions"]:
        check(types, "Question", q, "Quiz.questions[]")
        check(types, "SourceRef", q["source"], "Question.source")
    check(types, "Quiz", client.get(f"/api/quizzes/{quiz['id']}", headers=headers).json(), "GET quiz")

    graded = client.post(
        f"/api/quizzes/{quiz['id']}/grade", json={"answers": {"q0": "x"}}, headers={**headers, "X-Api-Key": "k"}
    ).json()
    check(types, "GradeResult", graded)
    for item in graded["results"]:
        check(types, "QuestionResult", item, "GradeResult.results[]")
        for kp in item.get("keyPoints") or []:
            check(types, "KeyPoint", kp)
    for t in graded["topics"]:
        check(types, "TopicScore", t)

    for s in client.get("/api/quizzes", headers=headers).json():
        check(types, "QuizSummary", s)


def test_question_types_and_statuses_are_ones_the_page_knows(client, types):  # noqa: F811
    text = TYPES.read_text()
    statuses = set(re.findall(r"'(\w+)'", re.search(r"status: ([^\n]+)", text).group(1)))
    headers, _ = join(client)
    quiz = make_quiz(client, headers, [make_doc(client, headers)["id"]])
    graded = client.post(
        f"/api/quizzes/{quiz['id']}/grade", json={"answers": {"q0": "x"}}, headers={**headers, "X-Api-Key": "k"}
    ).json()
    assert {r["status"] for r in graded["results"]} <= statuses
    q_types = set(re.findall(r"'(\w+)'", re.search(r"export type QuestionType = ([^\n]+)", text).group(1)))
    assert {q["type"] for q in quiz["questions"]} <= q_types


def test_every_error_code_the_server_uses_is_known_to_the_page(types):
    """Codes from app/api/errors.py and the upload stream must appear in the page's ApiErrorCode union."""
    from app.api import errors

    text = TYPES.read_text()
    union = re.search(r"export type ApiErrorCode =(.*?)\n\n", text, re.S).group(1)
    known = set(re.findall(r"'(\w+)'", union))
    source = (Path(__file__).resolve().parent.parent / "app" / "api" / "routes_documents.py").read_text()
    stream_codes = set(re.findall(r'"code": "(\w+)"', source)) | set(re.findall(r'code = "(\w+)"', source))
    server = (set(errors.STATUS) | stream_codes) - {"not_implemented", "forbidden", "invalid_request"}
    assert server - known == set(), f"the page does not know: {sorted(server - known)}"


def test_document_stream_events_match_the_client_parser(client):  # noqa: F811
    headers, _ = join(client)
    r = client.post("/api/documents", files={"file": ("n.md", b"# T\n" + b"Cells divide. " * 50)}, headers=headers)
    lines = [json.loads(x) for x in r.text.splitlines()]
    assert lines[-1]["event"] == "done" and set(lines[-1]) == {"event", "document"}
    bad = client.post("/api/documents", files={"file": ("x.mp3", b"ID3")}, headers=headers)
    err = json.loads(bad.text.splitlines()[-1])
    assert set(err) == {"event", "code", "message"}
