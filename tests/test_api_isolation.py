"""One account must never reach another account's data. This file checks it route by route, and fails when a
route is added without being covered here."""

import json

import pytest
from fastapi.testclient import TestClient

from app import config
from app.api import deps, llm_factory
from app.api.main import create_app
from app.auth import tokens
from tests.test_api import GOOD, join, make_doc, make_quiz

SECRET_A = b"# Secrets\n" + b"The vault of account A holds zebrafish blueprint ALPHA-7734. " * 40
SECRET_B = b"# Garden\n" + b"Account B grows tomatoes and basil in raised beds. " * 40


class SpyLLM:
    """Records every prompt, so a test can prove another account's text was never sent to the model."""

    max_concurrency = 1

    def __init__(self):
        self.prompts = []

    def generate(self, prompt, *, schema=None, **_):
        self.prompts.append(prompt)
        if "Evaluate the student" in prompt:
            return '{"score": 0.5, "feedback": "ok"}'
        return json.dumps(
            {
                "questions": [
                    {"question": f"Q{len(self.prompts)}-{i}?", "answer": "a", "topic": "T", "chunk": 1}
                    for i in range(5)
                ]
            }
        )


@pytest.fixture
def world(tmp_path, monkeypatch, store):
    """Account A has a file, a graded quiz and a second device; account B has a file of its own."""
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "iso.db")
    monkeypatch.setattr(config, "AUTH_SECRET", "s" * 40)
    monkeypatch.setattr(config, "COOKIE_SECURE", False)
    spy = SpyLLM()
    monkeypatch.setattr(llm_factory, "make_llm", lambda creds: spy)
    monkeypatch.setattr(llm_factory, "make_vision", lambda creds: None)
    deps.limiter.reset()
    with TestClient(create_app(), base_url="http://testserver") as client:
        a, _ = join(client, "alice@example.com")
        doc_a = make_doc(client, a, name="alice-secrets.md", data=SECRET_A)
        quiz_a = make_quiz(client, a, [doc_a["id"]])
        client.post(
            f"/api/quizzes/{quiz_a['id']}/grade", json={"answers": {"q0": "x"}}, headers={**a, "X-Api-Key": "k"}
        )
        device = TestClient(client.app, base_url="http://testserver")
        device.post("/api/auth/login", json={"email": "alice@example.com", "password": GOOD})
        a_session = next(s["id"] for s in client.get("/api/me/sessions", headers=a).json() if not s["current"])
        b, _ = join(client, "bob@example.com")
        doc_b = make_doc(client, b, name="bob-garden.md", data=SECRET_B)
        spy.prompts.clear()
        yield {
            "client": client,
            "a": a,
            "b": b,
            "doc_a": doc_a,
            "doc_b": doc_b,
            "quiz_a": quiz_a,
            "spy": spy,
            "a_session": a_session,
        }
    deps.limiter.reset()


def hb(w):
    return {**w["b"], "X-Api-Key": "k"}


class TestBobCannotReachAlice:
    def test_cannot_see_her_files(self, world):
        c = world["client"]
        assert [d["id"] for d in c.get("/api/documents", headers=world["b"]).json()] == [world["doc_b"]["id"]]

    def test_cannot_delete_her_file(self, world):
        c = world["client"]
        assert c.delete(f"/api/documents/{world['doc_a']['id']}", headers=world["b"]).status_code == 404
        assert [d["id"] for d in c.get("/api/documents", headers=world["a"]).json()] == [world["doc_a"]["id"]]

    def test_cannot_make_a_quiz_from_her_file_even_mixed_with_his_own(self, world):
        c = world["client"]
        for ids in ([world["doc_a"]["id"]], [world["doc_b"]["id"], world["doc_a"]["id"]]):
            r = c.post("/api/quizzes", json={"documentIds": ids, "count": 5, "type": "short"}, headers=hb(world))
            assert (r.status_code, r.json()["code"]) == (404, "not_found")
        assert world["spy"].prompts == []  # the model was never called

    def test_her_text_never_reaches_the_model_for_his_quiz(self, world):
        c = world["client"]
        make_quiz(c, world["b"], [world["doc_b"]["id"]])
        prompts = " ".join(world["spy"].prompts)
        assert "tomatoes" in prompts and "ALPHA-7734" not in prompts and "zebrafish" not in prompts

    def test_cannot_open_grade_or_read_her_quiz(self, world):
        c, q = world["client"], world["quiz_a"]["id"]
        assert c.get(f"/api/quizzes/{q}", headers=world["b"]).status_code == 404
        assert c.post(f"/api/quizzes/{q}/grade", json={"answers": {}}, headers=hb(world)).status_code == 404
        assert c.get(f"/api/quizzes/{q}/result", headers=world["b"]).status_code == 404

    def test_her_quizzes_are_not_in_his_list(self, world):
        assert world["client"].get("/api/quizzes", headers=world["b"]).json() == []

    def test_cannot_chat_over_her_file(self, world):
        r = world["client"].post(
            "/api/chat", json={"documentIds": [world["doc_a"]["id"]], "message": "hi"}, headers=world["b"]
        )
        assert r.status_code == 404

    def test_a_missing_id_and_her_id_look_identical(self, world):
        c = world["client"]
        mine_missing = c.get("/api/quizzes/not-a-real-id", headers=world["b"])
        hers = c.get(f"/api/quizzes/{world['quiz_a']['id']}", headers=world["b"])
        assert (mine_missing.status_code, mine_missing.json()) == (hers.status_code, hers.json())

    def test_cannot_sign_out_her_devices_or_list_them(self, world):
        c = world["client"]
        assert c.delete(f"/api/me/sessions/{world['a_session']}", headers=world["b"]).status_code == 404
        assert all(s["id"] != world["a_session"] for s in c.get("/api/me/sessions", headers=world["b"]).json())
        assert c.delete("/api/me/sessions/others", headers=world["b"]).json() == {"signedOut": 0}
        assert c.get("/api/me", headers=world["a"]).status_code == 200  # she is still signed in

    def test_his_export_holds_nothing_of_hers(self, world):
        text = world["client"].get("/api/me/export", headers=world["b"]).text
        for marker in ("alice", "alice-secrets", world["quiz_a"]["id"], world["doc_a"]["id"], "ALPHA-7734"):
            assert marker not in text

    def test_deleting_his_account_leaves_hers_whole(self, world, store):
        c = world["client"]
        assert c.request("DELETE", "/api/me", json={"email": "bob@example.com"}, headers=world["b"]).status_code == 204
        assert store.get_all_chunks(world["doc_a"]["id"]) and not store.get_all_chunks(world["doc_b"]["id"])
        assert c.get(f"/api/quizzes/{world['quiz_a']['id']}", headers=world["a"]).status_code == 200
        assert len(c.get("/api/documents", headers=world["a"]).json()) == 1

    def test_a_token_naming_his_id_with_her_session_is_refused(self, world):
        c = world["client"]
        bob_id = c.get("/api/me", headers=world["b"]).json()["id"]
        # Both are validly signed by the server's key; only the pairing is wrong.
        forged = tokens.issue_access_token(bob_id, world["a_session"])
        assert c.get("/api/me", headers={"Authorization": f"Bearer {forged}"}).status_code == 401

    def test_his_password_change_does_not_touch_her_sessions(self, world):
        c = world["client"]
        c.post("/api/me/password", json={"current": GOOD, "new": "bob has a new passphrase"}, headers=world["b"])
        assert c.get("/api/me", headers=world["a"]).status_code == 200
        assert any(s["id"] == world["a_session"] for s in c.get("/api/me/sessions", headers=world["a"]).json())


# Routes that work without being signed in. Everything else must answer 401 to an anonymous caller.
PUBLIC = {
    ("GET", "/api/health"),
    ("POST", "/api/auth/signup"),
    ("POST", "/api/auth/login"),
    ("POST", "/api/auth/guest"),
    ("POST", "/api/auth/refresh"),
    ("POST", "/api/auth/logout"),
}


def all_routes(app):
    paths = app.openapi()["paths"]
    return sorted((m.upper(), p) for p, methods in paths.items() for m in methods)


def test_every_route_that_is_not_public_refuses_anonymous_callers(world):
    client = world["client"]
    checked = []
    for method, path in all_routes(client.app):
        if (method, path) in PUBLIC:
            continue
        url = path.replace("{doc_id}", "x").replace("{quiz_id}", "x").replace("{session_id}", "x")
        r = client.request(method, url, json={})
        assert r.status_code == 401, f"{method} {path} answered {r.status_code} without a token"
        checked.append((method, path))
    assert len(checked) >= 15  # guards against the loop silently checking nothing


def test_every_route_is_known_to_this_file(world):
    """Adding a route fails this test until it is listed here and has an isolation test above (or is public)."""
    known = PUBLIC | {
        ("GET", "/api/me"),
        ("POST", "/api/me/password"),
        ("GET", "/api/me/sessions"),
        ("DELETE", "/api/me/sessions/others"),
        ("DELETE", "/api/me/sessions/{session_id}"),
        ("GET", "/api/me/export"),
        ("DELETE", "/api/me"),
        ("POST", "/api/auth/upgrade"),
        ("POST", "/api/documents"),
        ("POST", "/api/documents/url"),
        ("GET", "/api/documents"),
        ("DELETE", "/api/documents/{doc_id}"),
        ("POST", "/api/quizzes"),
        ("GET", "/api/quizzes"),
        ("GET", "/api/quizzes/{quiz_id}"),
        ("POST", "/api/quizzes/{quiz_id}/grade"),
        ("GET", "/api/quizzes/{quiz_id}/result"),
        ("POST", "/api/chat"),
    }
    assert set(all_routes(world["client"].app)) == known
