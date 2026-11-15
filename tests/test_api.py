"""The HTTP API: sign-up and sign-in, tokens and cookies, guests, documents, quizzes, and that one
account can never reach another account's data."""

import json
from contextlib import closing
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from fpdf import FPDF

from app import config
from app.api import deps, llm_factory
from app.api.main import create_app
from app.auth import sessions
from app.database import _connect

REAL_MAKE_LLM = llm_factory.make_llm  # the fixture below swaps the module attribute
GOOD = "correct horse battery"
ORIGIN = "http://localhost:5173"
CSRF = {"Origin": ORIGIN, "X-Requested-With": "quizforge"}
NOTES = ("# Cells\n" + "Mitochondria produce energy in the cell. " * 40).encode()


class QuizLLM:
    """Writes five questions per call and grades short answers as half right."""

    max_concurrency = 1

    def __init__(self):
        self.calls = 0

    def generate(self, prompt, *, schema=None, **_):
        self.calls += 1
        if "Evaluate the student" in prompt:
            return '{"score": 0.5, "feedback": "Half right"}'
        return json.dumps(
            {
                "questions": [
                    {"question": f"Q{self.calls}-{i}?", "answer": "an answer", "topic": "Cells", "chunk": 1}
                    for i in range(5)
                ]
            }
        )


class Vision:
    max_concurrency = 1

    def generate_from_file(self, prompt, data, mime, **_):
        return "Cells\n| a | b |\n|---|---|\n| 1 | 2 |\n[Figure: a cell]"


def blank_pdf(pages=2):
    pdf = FPDF()
    for _ in range(pages):
        pdf.add_page()
    return bytes(pdf.output())


@pytest.fixture
def client(tmp_path, monkeypatch, store):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "api.db")
    monkeypatch.setattr(config, "AUTH_SECRET", "s" * 40)
    monkeypatch.setattr(config, "COOKIE_SECURE", False)
    monkeypatch.setattr(config, "APP_ORIGIN", ORIGIN)
    monkeypatch.setattr(llm_factory, "make_llm", lambda creds: QuizLLM())
    monkeypatch.setattr(llm_factory, "make_vision", lambda creds: Vision() if creds.api_key else None)
    deps.limiter.reset()
    with TestClient(create_app(), base_url="http://testserver") as c:
        yield c
    deps.limiter.reset()


def join(client, email="a@example.com", password=GOOD, **extra):
    r = client.post("/api/auth/signup", json={"email": email, "password": password, **extra})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['accessToken']}"}, r


def upload(client, headers, name="notes.md", data=NOTES, key="k"):
    r = client.post("/api/documents", files={"file": (name, data)}, headers={**headers, "X-Api-Key": key})
    assert r.status_code == 200, r.text
    return [json.loads(line) for line in r.text.splitlines()]


def make_doc(client, headers, **kw):
    return upload(client, headers, **kw)[-1]["document"]


def make_quiz(client, headers, doc_ids):
    r = client.post(
        "/api/quizzes",
        json={"documentIds": doc_ids, "count": 5, "type": "short"},
        headers={**headers, "X-Api-Key": "k"},
    )
    assert r.status_code == 200, r.text
    return r.json()


class TestSignUpAndSignIn:
    def test_signup_returns_a_token_and_a_protected_cookie(self, client):
        headers, r = join(client)
        body = r.json()
        assert body["user"]["email"] == "a@example.com" and body["user"]["isGuest"] is False
        assert body["expiresIn"] == config.ACCESS_TTL_MIN * 60
        cookie = r.headers["set-cookie"]
        assert (
            "qf_refresh=" in cookie and "HttpOnly" in cookie and "SameSite=lax" in cookie and "Path=/api/auth" in cookie
        )
        assert "Max-Age" in cookie  # "keep me signed in" is the default
        assert "refresh" not in json.dumps(body).lower()  # the refresh token never appears in the body
        assert client.get("/api/me", headers=headers).json()["email"] == "a@example.com"

    def test_without_keep_me_signed_in_the_cookie_ends_with_the_browser(self, client):
        _, r = join(client, remember=False)
        assert "Max-Age" not in r.headers["set-cookie"] and "expires" not in r.headers["set-cookie"].lower()

    def test_weak_password_and_bad_email_and_duplicate(self, client):
        r = client.post("/api/auth/signup", json={"email": "a@example.com", "password": "short"})
        assert (r.status_code, r.json()["code"]) == (400, "weak_password")
        r = client.post("/api/auth/signup", json={"email": "not-an-email", "password": GOOD})
        assert r.json()["code"] == "invalid_email"
        join(client)
        r = client.post("/api/auth/signup", json={"email": "A@Example.com", "password": GOOD})
        assert (r.status_code, r.json()["code"]) == (409, "email_taken")

    def test_the_password_is_stored_hashed(self, client):
        join(client)
        with closing(_connect()) as conn:
            stored = conn.execute("SELECT password_hash FROM users").fetchone()[0]
        assert stored.startswith("$argon2id$") and GOOD not in stored

    def test_login_works_and_wrong_email_and_wrong_password_look_the_same(self, client):
        join(client)
        ok = client.post("/api/auth/login", json={"email": "A@example.com", "password": GOOD})
        assert ok.status_code == 200
        wrong_pw = client.post("/api/auth/login", json={"email": "a@example.com", "password": "wrong password!!"})
        no_user = client.post("/api/auth/login", json={"email": "nobody@example.com", "password": GOOD})
        assert wrong_pw.status_code == no_user.status_code == 401
        assert wrong_pw.json() == no_user.json() and wrong_pw.json()["code"] == "invalid_credentials"

    def test_sixth_login_attempt_in_a_minute_is_limited(self, client):
        join(client)
        for _ in range(5):
            assert (
                client.post(
                    "/api/auth/login", json={"email": "a@example.com", "password": "nope nope nope"}
                ).status_code
                == 401
            )
        r = client.post("/api/auth/login", json={"email": "a@example.com", "password": GOOD})
        assert r.status_code == 429 and r.json()["code"] == "rate_limited" and int(r.headers["retry-after"]) >= 1

    def test_signup_attempts_from_one_address_are_limited(self, client):
        for i in range(5):
            assert (
                client.post("/api/auth/signup", json={"email": f"u{i}@example.com", "password": GOOD}).status_code
                == 200
            )
        assert client.post("/api/auth/signup", json={"email": "u9@example.com", "password": GOOD}).status_code == 429

    def test_missing_forged_and_expired_tokens_get_401(self, client):
        for headers in (
            {},
            {"Authorization": "Bearer nonsense"},
            {"Authorization": "Basic abc"},
            {"Authorization": "Bearer "},
        ):
            r = client.get("/api/me", headers=headers)
            assert r.status_code == 401 and r.json()["code"] == "unauthorized"
        from app.auth import tokens

        old = tokens.issue_access_token("u", "s", now=datetime.now(timezone.utc).timestamp() - 7200)
        assert client.get("/api/me", headers={"Authorization": f"Bearer {old}"}).status_code == 401


class TestRefreshAndSignOut:
    def test_refresh_needs_the_same_origin_headers(self, client):
        join(client)
        assert client.post("/api/auth/refresh").status_code == 403
        assert (
            client.post(
                "/api/auth/refresh", headers={"Origin": "https://evil.example", "X-Requested-With": "quizforge"}
            ).json()["code"]
            == "csrf"
        )
        assert client.post("/api/auth/refresh", headers={"Origin": ORIGIN}).status_code == 403
        assert client.post("/api/auth/refresh", headers=CSRF).status_code == 200

    def test_refresh_rotates_the_cookie_and_returns_a_new_token(self, client):
        join(client)
        before = client.cookies.get("qf_refresh")
        r = client.post("/api/auth/refresh", headers=CSRF)
        assert r.status_code == 200 and r.json()["accessToken"]
        assert client.cookies.get("qf_refresh") != before

    def test_reusing_a_copied_cookie_ends_the_session(self, client, monkeypatch):
        monkeypatch.setattr(sessions, "GRACE_SECONDS", -1)
        headers, _ = join(client)
        stolen = client.cookies.get("qf_refresh")
        client.post("/api/auth/refresh", headers=CSRF)  # the real owner refreshes
        thief = TestClient(client.app, base_url="http://testserver")
        thief.cookies.set("qf_refresh", stolen, path="/api/auth")
        r = thief.post("/api/auth/refresh", headers=CSRF)
        assert r.status_code == 401
        assert "Max-Age=0" in r.headers["set-cookie"]  # the cookie is cleared
        # the owner's session is dead too: copying a token costs the owner, so they will notice and sign in again
        assert client.post("/api/auth/refresh", headers=CSRF).status_code == 401
        assert client.get("/api/me", headers=headers).status_code == 401

    def test_two_quick_refreshes_do_not_sign_you_out(self, client):
        join(client)
        old = client.cookies.get("qf_refresh")
        assert client.post("/api/auth/refresh", headers=CSRF).status_code == 200
        twin = TestClient(client.app, base_url="http://testserver")
        twin.cookies.set("qf_refresh", old, path="/api/auth")  # a second tab that still has the old cookie
        assert twin.post("/api/auth/refresh", headers=CSRF).status_code == 200
        assert client.post("/api/auth/refresh", headers=CSRF).status_code == 200

    def test_no_cookie_means_signed_out(self, client):
        r = client.post("/api/auth/refresh", headers=CSRF)
        assert r.status_code == 401 and r.json()["code"] == "unauthorized"

    def test_logout_revokes_the_session_straight_away(self, client):
        headers, _ = join(client)
        assert client.get("/api/me", headers=headers).status_code == 200
        r = client.post("/api/auth/logout", headers=CSRF)
        assert r.status_code == 204 and "Max-Age=0" in r.headers["set-cookie"]
        assert (
            client.get("/api/me", headers=headers).status_code == 401
        )  # the token itself would still be valid for minutes
        assert client.post("/api/auth/logout").status_code == 403


class TestGuests:
    def test_guest_can_use_the_app_and_has_an_expiry(self, client):
        r = client.post("/api/auth/guest")
        assert r.status_code == 200 and r.json()["user"]["isGuest"] and r.json()["user"]["expiresAt"]
        headers = {"Authorization": f"Bearer {r.json()['accessToken']}"}
        assert client.get("/api/me", headers=headers).json()["email"] is None

    def test_guest_file_limit(self, client, monkeypatch):
        monkeypatch.setattr(config, "GUEST_MAX_FILES", 2)
        headers = {"Authorization": "Bearer " + client.post("/api/auth/guest").json()["accessToken"]}
        make_doc(client, headers, name="a.md")
        make_doc(client, headers, name="b.md")
        r = client.post("/api/documents", files={"file": ("c.md", NOTES)}, headers=headers)
        assert r.status_code == 403 and r.json()["code"] == "guest_limit"

    def test_guest_size_limit(self, client, monkeypatch):
        monkeypatch.setattr(config, "GUEST_MAX_UPLOAD_MB", 0.0001)
        headers = {"Authorization": "Bearer " + client.post("/api/auth/guest").json()["accessToken"]}
        assert (
            client.post("/api/documents", files={"file": ("big.md", NOTES)}, headers=headers).json()["code"]
            == "guest_limit"
        )

    def test_an_expired_guest_is_signed_out(self, client):
        r = client.post("/api/auth/guest")
        headers = {"Authorization": f"Bearer {r.json()['accessToken']}"}
        with closing(_connect()) as conn:
            conn.execute(
                "UPDATE users SET expires_at = ?", ((datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),)
            )
            conn.commit()
        assert client.get("/api/me", headers=headers).status_code == 401

    def test_upgrade_keeps_files_and_quizzes_and_the_same_id(self, client):
        r = client.post("/api/auth/guest")
        headers = {"Authorization": f"Bearer {r.json()['accessToken']}"}
        guest_id = r.json()["user"]["id"]
        doc = make_doc(client, headers)
        quiz = make_quiz(client, headers, [doc["id"]])
        up = client.post("/api/auth/upgrade", json={"email": "me@example.com", "password": GOOD}, headers=headers)
        assert up.status_code == 200 and up.json()["user"] == {
            "id": guest_id,
            "email": "me@example.com",
            "isGuest": False,
            "expiresAt": None,
        }
        new_headers = {"Authorization": f"Bearer {up.json()['accessToken']}"}
        assert [d["id"] for d in client.get("/api/documents", headers=new_headers).json()] == [doc["id"]]
        assert client.get(f"/api/quizzes/{quiz['id']}", headers=new_headers).status_code == 200
        assert client.post("/api/auth/login", json={"email": "me@example.com", "password": GOOD}).status_code == 200

    def test_upgrade_rules(self, client):
        headers, _ = join(client)
        r = client.post("/api/auth/upgrade", json={"email": "x@example.com", "password": GOOD}, headers=headers)
        assert r.json()["code"] == "not_guest"
        guest = {"Authorization": "Bearer " + client.post("/api/auth/guest").json()["accessToken"]}
        taken = client.post("/api/auth/upgrade", json={"email": "a@example.com", "password": GOOD}, headers=guest)
        assert taken.status_code == 409
        weak = client.post("/api/auth/upgrade", json={"email": "z@example.com", "password": "weak"}, headers=guest)
        assert weak.json()["code"] == "weak_password"


class TestAccount:
    def test_change_password_signs_out_other_devices(self, client):
        headers, _ = join(client)
        other = TestClient(client.app, base_url="http://testserver")
        other_headers = {
            "Authorization": "Bearer "
            + other.post("/api/auth/login", json={"email": "a@example.com", "password": GOOD}).json()["accessToken"]
        }
        r = client.post("/api/me/password", json={"current": GOOD, "new": "a brand new passphrase"}, headers=headers)
        assert r.status_code == 204
        assert client.get("/api/me", headers=headers).status_code == 200  # this device stays signed in
        assert other.get("/api/me", headers=other_headers).status_code == 401
        assert client.post("/api/auth/login", json={"email": "a@example.com", "password": GOOD}).status_code == 401
        assert (
            client.post(
                "/api/auth/login", json={"email": "a@example.com", "password": "a brand new passphrase"}
            ).status_code
            == 200
        )

    def test_change_password_checks_the_current_one_and_the_policy(self, client):
        headers, _ = join(client)
        assert (
            client.post(
                "/api/me/password", json={"current": "not it at all", "new": "another passphrase"}, headers=headers
            ).json()["code"]
            == "wrong_password"
        )
        assert (
            client.post("/api/me/password", json={"current": GOOD, "new": "short"}, headers=headers).json()["code"]
            == "weak_password"
        )
        guest = {"Authorization": "Bearer " + client.post("/api/auth/guest").json()["accessToken"]}
        assert (
            client.post("/api/me/password", json={"current": "x", "new": "another passphrase"}, headers=guest).json()[
                "code"
            ]
            == "no_password"
        )

    def test_devices_list_and_sign_out(self, client):
        headers, _ = join(client)
        second = TestClient(client.app, base_url="http://testserver")
        second.post(
            "/api/auth/login",
            json={"email": "a@example.com", "password": GOOD},
            headers={"User-Agent": "Safari on iPhone"},
        )
        devices = client.get("/api/me/sessions", headers=headers).json()
        assert len(devices) == 2 and sum(d["current"] for d in devices) == 1
        other = next(d for d in devices if not d["current"])
        assert other["userAgent"] == "Safari on iPhone"
        assert client.delete(f"/api/me/sessions/{other['id']}", headers=headers).status_code == 204
        assert len(client.get("/api/me/sessions", headers=headers).json()) == 1
        assert client.delete("/api/me/sessions/does-not-exist", headers=headers).status_code == 404

    def test_sign_out_everywhere_else(self, client):
        headers, _ = join(client)
        for _ in range(2):
            TestClient(client.app, base_url="http://testserver").post(
                "/api/auth/login", json={"email": "a@example.com", "password": GOOD}
            )
        assert client.delete("/api/me/sessions/others", headers=headers).json() == {"signedOut": 2}
        assert client.get("/api/me", headers=headers).status_code == 200

    def test_me_counts_and_export(self, client):
        headers, _ = join(client)
        doc = make_doc(client, headers)
        quiz = make_quiz(client, headers, [doc["id"]])
        client.post(
            f"/api/quizzes/{quiz['id']}/grade",
            json={"answers": {"q0": "an answer"}},
            headers={**headers, "X-Api-Key": "k"},
        )
        me = client.get("/api/me", headers=headers).json()
        assert me["counts"]["files"] == 1 and me["counts"]["quizzes"] == 1
        data = client.get("/api/me/export", headers=headers)
        assert "attachment" in data.headers["content-disposition"]
        body = data.json()
        assert body["account"]["email"] == "a@example.com" and len(body["files"]) == 1
        assert len(body["quizzes"]) == 1 and len(body["attempts"]) == 1
        assert "password" not in data.text.lower()

    def test_deleting_the_account_removes_everything(self, client, store):
        headers, _ = join(client)
        doc = make_doc(client, headers)
        quiz = make_quiz(client, headers, [doc["id"]])
        client.post(
            f"/api/quizzes/{quiz['id']}/grade",
            json={"answers": {"q0": "an answer"}},
            headers={**headers, "X-Api-Key": "k"},
        )
        assert store.get_all_chunks(doc["id"])
        r = client.request("DELETE", "/api/me", json={"email": "wrong@example.com"}, headers=headers)
        assert r.json()["code"] == "confirmation_mismatch" and client.get("/api/me", headers=headers).status_code == 200
        assert client.request("DELETE", "/api/me", json={"email": "A@example.com"}, headers=headers).status_code == 204
        assert client.get("/api/me", headers=headers).status_code == 401
        assert store.get_all_chunks(doc["id"]) == []
        with closing(_connect()) as conn:
            for table in ("users", "sessions", "docs", "quizzes", "attempts", "results"):
                assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0, table
        assert client.post("/api/auth/login", json={"email": "a@example.com", "password": GOOD}).status_code == 401


class TestDocuments:
    def test_upload_streams_a_done_line(self, client):
        headers, _ = join(client)
        events = upload(client, headers)
        assert events[-1]["event"] == "done"
        doc = events[-1]["document"]
        assert doc["name"] == "notes.md" and doc["method"] == "text" and doc["pages"] is None
        assert client.get("/api/documents", headers=headers).json() == [doc]

    def test_scan_reports_progress_and_what_was_kept(self, client):
        headers, _ = join(client)
        events = upload(client, headers, name="lecture scan.pdf", data=blank_pdf(3))
        progress = [e for e in events if e["event"] == "progress"]
        assert [p["page"] for p in progress] == [1, 2, 3] and progress[-1]["pages"] == 3
        doc = events[-1]["document"]
        assert (doc["method"], doc["pages"], doc["tables"], doc["figures"]) == ("ocr", 3, 3, 3)

    def test_scan_without_a_gemini_key_says_why(self, client):
        headers, _ = join(client)
        events = upload(client, headers, name="scan.pdf", data=blank_pdf(2), key="")
        assert events[-1]["event"] == "error" and events[-1]["code"] == "unreadable"

    def test_image_needs_gemini(self, client):
        headers, _ = join(client)
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
        assert upload(client, headers, name="d.png", data=png, key="")[-1]["code"] == "needs_gemini"
        assert upload(client, headers, name="d.png", data=png, key="k")[-1]["document"]["method"] == "image"

    def test_long_scan_is_refused_before_any_page_is_read(self, client, monkeypatch):
        monkeypatch.setattr(config, "OCR_MAX_PAGES", 2)
        headers, _ = join(client)
        assert upload(client, headers, name="scan.pdf", data=blank_pdf(3))[-1]["code"] == "ocr_limit"

    def test_unsupported_type(self, client):
        headers, _ = join(client)
        assert upload(client, headers, name="song.mp3", data=b"ID3\x00\x00")[-1]["code"] == "unsupported"

    def test_failed_upload_leaves_no_document_behind(self, client):
        headers, _ = join(client)
        upload(client, headers, name="song.mp3", data=b"ID3\x00\x00")
        assert client.get("/api/documents", headers=headers).json() == []

    def test_add_a_web_page(self, client, monkeypatch):
        import app.loaders as loaders

        monkeypatch.setattr(
            loaders,
            "_http_get",
            lambda url, limit, timeout: (("<p>Cells divide by mitosis. </p>" * 40).encode(), "text/html"),
        )
        monkeypatch.setattr(loaders.socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))])
        headers, _ = join(client)
        r = client.post("/api/documents/url", json={"url": "https://example.com/bio"}, headers=headers)
        assert json.loads(r.text.splitlines()[-1])["document"]["name"] == "example.com/bio"

    def test_delete_a_document(self, client, store):
        headers, _ = join(client)
        doc = make_doc(client, headers)
        assert client.delete(f"/api/documents/{doc['id']}", headers=headers).status_code == 204
        assert store.get_all_chunks(doc["id"]) == []
        assert client.delete(f"/api/documents/{doc['id']}", headers=headers).status_code == 404

    def test_documents_need_sign_in(self, client):
        assert client.post("/api/documents", files={"file": ("a.md", NOTES)}).status_code == 401
        assert client.get("/api/documents").status_code == 401


class TestQuizzes:
    def test_make_take_and_grade(self, client):
        headers, _ = join(client)
        doc = make_doc(client, headers)
        quiz = make_quiz(client, headers, [doc["id"]])
        assert len(quiz["questions"]) == 5 and quiz["questions"][0]["id"] == "q0"
        assert quiz["questions"][0]["source"] == {"file": "notes.md", "page": 0}
        graded = client.post(
            f"/api/quizzes/{quiz['id']}/grade",
            json={"answers": {"q0": "my answer", "q1": ""}},
            headers={**headers, "X-Api-Key": "k"},
        ).json()
        assert (
            graded["total"] == 5
            and graded["results"][0]["status"] == "partial"
            and graded["results"][0]["score"] == 0.5
        )
        assert graded["results"][1]["status"] == "incorrect"  # blank answer
        assert graded["average"] == pytest.approx(0.1)
        assert client.get(f"/api/quizzes/{quiz['id']}/result", headers=headers).json() == graded
        listing = client.get("/api/quizzes", headers=headers).json()
        assert (
            listing[0]["id"] == quiz["id"]
            and listing[0]["lastScore"] == pytest.approx(0.1)
            and listing[0]["questionCount"] == 5
        )

    def test_result_before_grading_is_not_found(self, client):
        headers, _ = join(client)
        quiz = make_quiz(client, headers, [make_doc(client, headers)["id"]])
        assert client.get(f"/api/quizzes/{quiz['id']}/result", headers=headers).status_code == 404

    def test_request_is_validated(self, client):
        headers, _ = join(client)
        doc = make_doc(client, headers)
        for bad in (
            {"documentIds": [], "count": 5},
            {"documentIds": [doc["id"]], "count": 4},
            {"documentIds": [doc["id"]], "count": 26},
            {"documentIds": [doc["id"]], "count": 5, "type": "essay"},
        ):
            r = client.post("/api/quizzes", json=bad, headers=headers)
            assert (r.status_code, r.json()["code"]) == (400, "invalid_request")

    def test_the_real_factory_asks_for_a_gemini_key(self):
        from app.api.errors import ApiProblem

        with pytest.raises(ApiProblem) as err:
            REAL_MAKE_LLM(deps.ModelCredentials("Gemini", ""))
        assert err.value.code == "needs_gemini"

    def test_latest_attempt_is_the_one_shown(self, client):
        headers, _ = join(client)
        quiz = make_quiz(client, headers, [make_doc(client, headers)["id"]])
        h = {**headers, "X-Api-Key": "k"}
        client.post(f"/api/quizzes/{quiz['id']}/grade", json={"answers": {}}, headers=h)
        client.post(f"/api/quizzes/{quiz['id']}/grade", json={"answers": {f"q{i}": "x" for i in range(5)}}, headers=h)
        assert client.get(f"/api/quizzes/{quiz['id']}/result", headers=headers).json()["average"] == pytest.approx(0.5)


class TestApiBasics:
    def test_health_and_headers(self, client):
        r = client.get("/api/health")
        assert r.json() == {"status": "ok"}
        assert r.headers["x-content-type-options"] == "nosniff" and r.headers["x-frame-options"] == "DENY"
        headers, signup = join(client)
        assert signup.headers["cache-control"] == "no-store"

    def test_cors_allows_only_the_app_origin(self, client):
        good = client.options("/api/me", headers={"Origin": ORIGIN, "Access-Control-Request-Method": "GET"})
        assert (
            good.headers["access-control-allow-origin"] == ORIGIN
            and good.headers["access-control-allow-credentials"] == "true"
        )
        bad = client.options(
            "/api/me", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"}
        )
        assert "access-control-allow-origin" not in bad.headers

    def test_errors_share_one_shape(self, client):
        r = client.get("/api/nope")
        assert r.status_code == 404 and set(r.json()) == {"code", "message"}
        assert set(client.post("/api/auth/login", json={}).json()) == {"code", "message"}

    def test_chat_checks_ownership_then_says_it_is_not_built(self, client):
        headers, _ = join(client)
        assert (
            client.post("/api/chat", json={"documentIds": ["nope"], "message": "hi"}, headers=headers).status_code
            == 404
        )
        doc = make_doc(client, headers)
        r = client.post("/api/chat", json={"documentIds": [doc["id"]], "message": "hi"}, headers=headers)
        assert (r.status_code, r.json()["code"]) == (501, "not_implemented")

    def test_client_address_from_proxy_header_only_counts_trusted_hops(self, monkeypatch):
        from types import SimpleNamespace

        monkeypatch.setattr(config, "TRUSTED_PROXY_HOPS", 1)
        req = SimpleNamespace(
            headers={"x-forwarded-for": "6.6.6.6, 203.0.113.9"}, client=SimpleNamespace(host="10.0.0.1")
        )
        assert deps.client_ip(req) == "203.0.113.9"  # the made-up left entry is ignored
        monkeypatch.setattr(config, "TRUSTED_PROXY_HOPS", 0)
        assert deps.client_ip(req) == "10.0.0.1"
