"""Passwords, tokens, login sessions, users and the rate limiter, with no HTTP in the way."""

import time
from datetime import datetime, timedelta, timezone

import jwt
import pytest

from app import config
from app.auth import passwords, sessions, tokens, users
from app.auth.ratelimit import RateLimiter

T0 = datetime(2026, 11, 14, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(config, "AUTH_SECRET", "x" * 40)
    return tmp_path / "auth.db"


class TestPasswords:
    def test_boundary_is_twelve(self):
        passwords.check_policy("x7!k q2#m 9zp")  # 13
        passwords.check_policy("zq7xkp2wvz94")  # exactly 12
        with pytest.raises(passwords.PasswordPolicyError):
            passwords.check_policy("zq7xkp2wvz9")  # 11
        with pytest.raises(passwords.PasswordPolicyError, match="at most"):
            passwords.check_policy("a b" * 50)  # 150

    def test_common_and_repeated_passwords_refused(self):
        for bad in ("password1234", "PASSWORD1234", "aaaaaaaaaaaaaa", "qwertyuiop12"):
            with pytest.raises(passwords.PasswordPolicyError, match="common"):
                passwords.check_policy(bad)

    def test_not_the_email(self):
        with pytest.raises(passwords.PasswordPolicyError, match="email"):
            passwords.check_policy("janedoe-example", "janedoe-example@example.com")

    def test_hash_verify_and_never_the_password_itself(self):
        h = passwords.hash_password("correct horse battery")
        assert h.startswith("$argon2id$") and "correct" not in h
        assert passwords.verify_password("correct horse battery", h)
        assert not passwords.verify_password("correct horse batterz", h)
        assert not passwords.verify_password("anything", "not a hash")
        assert passwords.hash_password("correct horse battery") != h  # salted

    def test_rehash_flag_for_an_old_hash(self):
        from argon2 import PasswordHasher

        weak = PasswordHasher(time_cost=1, memory_cost=8, parallelism=1).hash("some long password")
        assert passwords.needs_rehash(weak)
        assert not passwords.needs_rehash(passwords.hash_password("some long password"))


class TestTokens:
    def test_round_trip(self, db):
        claims = tokens.decode_access_token(tokens.issue_access_token("u1", "s1"))
        assert (claims.user_id, claims.session_id) == ("u1", "s1")

    def test_expired_token_refused(self, db):
        old = tokens.issue_access_token("u1", "s1", now=time.time() - 3600)
        with pytest.raises(tokens.InvalidToken):
            tokens.decode_access_token(old)

    def test_lifetime_follows_config(self, db, monkeypatch):
        monkeypatch.setattr(config, "ACCESS_TTL_MIN", 5)
        c = tokens.decode_access_token(tokens.issue_access_token("u", "s", now=time.time()))
        assert 290 <= c.expires_at - int(time.time()) <= 300

    def test_tampered_signature_refused(self, db):
        token = tokens.issue_access_token("u1", "s1")
        with pytest.raises(tokens.InvalidToken):
            tokens.decode_access_token(token[:-3] + ("AAA" if not token.endswith("AAA") else "BBB"))

    def test_alg_none_refused(self, db):
        claims = {
            "sub": "u1",
            "sid": "s1",
            "iat": int(time.time()),
            "exp": int(time.time()) + 600,
            "iss": tokens.ISSUER,
            "aud": tokens.AUDIENCE,
        }
        forged = jwt.encode(claims, key=None, algorithm="none")
        with pytest.raises(tokens.InvalidToken):
            tokens.decode_access_token(forged)

    def test_another_algorithm_refused(self, db):
        claims = {
            "sub": "u1",
            "sid": "s1",
            "iat": int(time.time()),
            "exp": int(time.time()) + 600,
            "iss": tokens.ISSUER,
            "aud": tokens.AUDIENCE,
        }
        other = jwt.encode(claims, "x" * 40, algorithm="HS512")
        with pytest.raises(tokens.InvalidToken):
            tokens.decode_access_token(other)

    def test_wrong_audience_issuer_secret_or_missing_claim_refused(self, db):
        good = {
            "sub": "u1",
            "sid": "s1",
            "iat": int(time.time()),
            "exp": int(time.time()) + 600,
            "iss": tokens.ISSUER,
            "aud": tokens.AUDIENCE,
        }
        for change in ({"aud": "other-service"}, {"iss": "someone-else"}):
            with pytest.raises(tokens.InvalidToken):
                tokens.decode_access_token(jwt.encode({**good, **change}, "x" * 40, algorithm="HS256"))
        with pytest.raises(tokens.InvalidToken):
            tokens.decode_access_token(jwt.encode(good, "y" * 40, algorithm="HS256"))
        without_sid = {k: v for k, v in good.items() if k != "sid"}
        with pytest.raises(tokens.InvalidToken):
            tokens.decode_access_token(jwt.encode(without_sid, "x" * 40, algorithm="HS256"))

    def test_garbage_refused(self, db):
        for junk in ("", "abc", "a.b.c"):
            with pytest.raises(tokens.InvalidToken):
                tokens.decode_access_token(junk)

    def test_production_needs_a_secret_and_it_must_be_long(self, monkeypatch):
        monkeypatch.setattr(config, "AUTH_SECRET", "")
        monkeypatch.setattr(config, "APP_ENV", "production")
        with pytest.raises(tokens.AuthConfigError):
            tokens.secret()
        monkeypatch.setattr(config, "AUTH_SECRET", "too-short")
        with pytest.raises(tokens.AuthConfigError, match="32"):
            tokens.secret()

    def test_development_without_a_secret_uses_a_throwaway_one(self, monkeypatch):
        monkeypatch.setattr(config, "AUTH_SECRET", "")
        monkeypatch.setattr(config, "APP_ENV", "development")
        assert len(tokens.secret()) >= 32 and tokens.secret() == tokens.secret()

    def test_refresh_tokens_are_random_and_only_their_hash_is_kept(self):
        a, b = tokens.new_refresh_token(), tokens.new_refresh_token()
        assert a != b and len(a) >= 40
        assert tokens.hash_token(a) != a and len(tokens.hash_token(a)) == 64


class TestUsers:
    def test_email_is_normalised_and_unique(self, db):
        u = users.create_user("  Siva@Example.COM ", "hash")
        assert u.email == "siva@example.com" and not u.is_guest
        with pytest.raises(users.EmailTaken):
            users.create_user("SIVA@example.com", "hash2")

    def test_bad_emails_refused(self, db):
        for bad in ("", "no-at-sign", "a@b", "a b@c.com", "x" * 250 + "@a.com"):
            with pytest.raises(users.InvalidEmail):
                users.create_user(bad, "h")

    def test_login_lookup_hides_malformed_and_guest_accounts(self, db):
        users.create_user("a@b.co", "H1")
        assert users.get_with_hash("A@B.co")[1] == "H1"
        assert users.get_with_hash("nope") is None
        assert users.get_with_hash("zzz@b.co") is None

    def test_guest_expires_after_the_configured_days(self, db, monkeypatch):
        monkeypatch.setattr(config, "GUEST_TTL_DAYS", 7)
        g = users.create_guest(now=T0)
        assert g.is_guest and g.email is None
        assert not users.is_expired(g, now=T0 + timedelta(days=6, hours=23))
        assert users.is_expired(g, now=T0 + timedelta(days=7))

    def test_upgrade_keeps_the_same_id_and_clears_expiry(self, db):
        g = users.create_guest()
        up = users.upgrade_guest(g.id, "me@x.io", "H")
        assert up.id == g.id and up.email == "me@x.io" and not up.is_guest and up.expires_at is None
        with pytest.raises(ValueError):
            users.upgrade_guest(g.id, "again@x.io", "H")  # no longer a guest

    def test_upgrade_to_a_taken_email_refused(self, db):
        users.create_user("taken@x.io", "H")
        g = users.create_guest()
        with pytest.raises(users.EmailTaken):
            users.upgrade_guest(g.id, "Taken@x.io", "H")
        assert users.get_user(g.id).is_guest  # nothing changed

    def test_expired_guests_are_deleted_with_their_data(self, db, store):
        from app.database import _connect
        from app.quizzes import save_quiz

        old, fresh = users.create_guest(now=T0), users.create_guest(now=T0 + timedelta(days=6))
        save_quiz(old.id, [{"question": "q"}])
        save_quiz(fresh.id, [{"question": "q"}])
        assert users.delete_expired_guests(now=T0 + timedelta(days=8)) == 1
        assert users.get_user(old.id) is None and users.get_user(fresh.id) is not None
        with _connect() as conn:
            assert conn.execute("SELECT COUNT(*) FROM quizzes WHERE owner = ?", (old.id,)).fetchone()[0] == 0
            assert conn.execute("SELECT COUNT(*) FROM quizzes WHERE owner = ?", (fresh.id,)).fetchone()[0] == 1


class TestSessions:
    def test_rotation_replaces_the_token(self, db):
        u = users.create_user("a@b.co", "H")
        sid, first, _ = sessions.create_session(u.id, remember=True, now=T0)
        r = sessions.rotate(first, now=T0 + timedelta(minutes=20))
        assert r.session_id == sid and r.user_id == u.id and r.new_token and r.new_token != first
        r2 = sessions.rotate(r.new_token, now=T0 + timedelta(minutes=40))
        assert r2.new_token and r2.new_token != r.new_token

    def test_only_a_hash_is_stored(self, db):
        from app.database import _connect

        u = users.create_user("a@b.co", "H")
        _, token, _ = sessions.create_session(u.id, remember=True)
        with _connect() as conn:
            stored = [r[0] for r in conn.execute("SELECT token_hash FROM sessions")]
        assert token not in stored and tokens.hash_token(token) in stored

    def test_reusing_an_old_token_revokes_the_whole_session(self, db):
        u = users.create_user("a@b.co", "H")
        sid, first, _ = sessions.create_session(u.id, remember=True, now=T0)
        newer = sessions.rotate(first, now=T0 + timedelta(minutes=20)).new_token
        # A copy of the first token is used a minute later, well after rotation.
        with pytest.raises(sessions.ReuseDetected):
            sessions.rotate(first, now=T0 + timedelta(minutes=21))
        assert not sessions.is_active(sid, u.id, now=T0 + timedelta(minutes=21))
        with pytest.raises(sessions.InvalidRefresh):  # even the legitimate newest token is now dead
            sessions.rotate(newer, now=T0 + timedelta(minutes=22))

    def test_a_quick_second_refresh_is_allowed_once_without_a_new_token(self, db):
        u = users.create_user("a@b.co", "H")
        sid, first, _ = sessions.create_session(u.id, remember=True, now=T0)
        sessions.rotate(first, now=T0 + timedelta(minutes=20))
        again = sessions.rotate(first, now=T0 + timedelta(minutes=20, seconds=3))
        assert again.session_id == sid and again.new_token is None
        assert sessions.is_active(sid, u.id, now=T0 + timedelta(minutes=21))

    def test_unknown_and_empty_tokens_refused(self, db):
        for t in ("", "nonsense"):
            with pytest.raises(sessions.InvalidRefresh):
                sessions.rotate(t)

    def test_expiry_differs_with_keep_me_signed_in(self, db, monkeypatch):
        monkeypatch.setattr(config, "REFRESH_TTL_DAYS", 30)
        monkeypatch.setattr(config, "SESSION_TTL_HOURS", 12)
        u = users.create_user("a@b.co", "H")
        _, long_token, _ = sessions.create_session(u.id, remember=True, now=T0)
        _, short_token, _ = sessions.create_session(u.id, remember=False, now=T0)
        assert sessions.rotate(long_token, now=T0 + timedelta(days=29)).new_token
        with pytest.raises(sessions.InvalidRefresh):
            sessions.rotate(short_token, now=T0 + timedelta(hours=13))
        with pytest.raises(sessions.InvalidRefresh):
            sessions.rotate(sessions.create_session(u.id, remember=True, now=T0)[1], now=T0 + timedelta(days=31))

    def test_revoke_checks_the_owner(self, db):
        a, b = users.create_user("a@b.co", "H"), users.create_user("b@b.co", "H")
        sid_a, tok_a, _ = sessions.create_session(a.id, remember=True)
        assert not sessions.revoke(sid_a, user_id=b.id)  # someone else's session
        assert sessions.is_active(sid_a, a.id)
        assert sessions.revoke(sid_a, user_id=a.id)
        with pytest.raises(sessions.InvalidRefresh):
            sessions.rotate(tok_a)

    def test_is_active_needs_the_matching_user(self, db):
        a, b = users.create_user("a@b.co", "H"), users.create_user("b@b.co", "H")
        sid, _, _ = sessions.create_session(a.id, remember=True)
        assert sessions.is_active(sid, a.id) and not sessions.is_active(sid, b.id)

    def test_sign_out_everywhere_else_keeps_this_one(self, db):
        u = users.create_user("a@b.co", "H")
        keep, _, _ = sessions.create_session(u.id, remember=True)
        other, _, _ = sessions.create_session(u.id, remember=True)
        assert sessions.revoke_others(u.id, keep) == 1
        assert sessions.is_active(keep, u.id) and not sessions.is_active(other, u.id)
        assert [s.id for s in sessions.list_sessions(u.id)] == [keep]


class TestRateLimiter:
    def test_blocks_after_the_limit_then_recovers(self):
        now = [0.0]
        rl = RateLimiter(clock=lambda: now[0])
        assert all(rl.allow("ip", 3, 60) for _ in range(3))
        assert not rl.allow("ip", 3, 60)
        assert rl.retry_after("ip", 60) >= 1
        now[0] = 61
        assert rl.allow("ip", 3, 60)

    def test_keys_are_independent(self):
        rl = RateLimiter(clock=lambda: 0.0)
        assert rl.allow("a", 1, 60) and not rl.allow("a", 1, 60)
        assert rl.allow("b", 1, 60)
