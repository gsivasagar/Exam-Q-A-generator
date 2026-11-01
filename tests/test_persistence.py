"""Schema migrations, saved quizzes, spaced repetition, exports and cleanup."""
import csv
import io
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app import config
from app.database import SCHEMA_VERSION, delete_owner_data, init_db, load_results, migrate, store_results
from app.exports import exam_pdf, to_anki_tsv, to_csv
from app.maintenance import cleanup_expired, expired_docs, forget_doc, register_doc
from app.quizzes import delete_quiz, list_quizzes, load_quiz, save_quiz
from app.srs import card_id, count_due, due_questions, next_state, record_results
from tests.conftest import FakeLLM, make_pdf_bytes

NOW = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def db(tmp_path):
    return tmp_path / "t.db"


def q(text="What is X?", answer="X is Y.", **kw):
    return {"question": text, "answer": answer, "type": "short", "topic": "T", "source": "n.pdf", "page": 2, **kw}


class TestMigrations:
    def test_fresh_database_gets_latest_schema(self, db):
        init_db(db)
        with closing(sqlite3.connect(db)) as c:
            assert c.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
            tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"results", "quizzes", "srs", "docs"} <= tables

    def test_legacy_database_is_upgraded_without_losing_data(self, db):
        with closing(sqlite3.connect(db)) as c:   # a database from before versioning existed
            c.execute("""CREATE TABLE results (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, session_id TEXT,
                         quiz_id TEXT, question TEXT, topic TEXT, student TEXT, answer TEXT, score REAL,
                         feedback TEXT)""")
            c.execute("INSERT INTO results (timestamp, session_id, quiz_id, question, score) "
                      "VALUES ('2026-01-01', 's', 'q', 'old question', 0.5)")
            c.commit()
        init_db(db)
        assert load_results("s", path=db).iloc[0]["question"] == "old question"
        with closing(sqlite3.connect(db)) as c:
            assert c.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION

    def test_idempotent(self, db):
        init_db(db)
        with closing(sqlite3.connect(db)) as c:
            assert migrate(c) == migrate(c) == SCHEMA_VERSION


class TestQuizzes:
    def test_save_load_list_delete(self, db):
        qid = save_quiz("alice", [q(), q("Q2?")], title="Biology", question_type="short", path=db)
        got = load_quiz("alice", qid, path=db)
        assert got["title"] == "Biology" and [x["question"] for x in got["questions"]] == ["What is X?", "Q2?"]
        assert [x["id"] for x in list_quizzes("alice", path=db)] == [qid]
        assert delete_quiz("alice", qid, path=db) and load_quiz("alice", qid, path=db) is None
        assert not delete_quiz("alice", qid, path=db)

    def test_owners_are_isolated(self, db):
        qid = save_quiz("alice", [q()], path=db)
        assert load_quiz("bob", qid, path=db) is None
        assert list_quizzes("bob", path=db) == [] and not delete_quiz("bob", qid, path=db)
        assert load_quiz("alice", qid, path=db) is not None

    def test_list_shows_attempts_and_average(self, db):
        qid = save_quiz("alice", [q()], path=db)
        store_results([{"question": "a", "answer": "b", "score": 1.0}], "alice", qid, path=db)
        store_results([{"question": "a", "answer": "b", "score": 0.5}], "alice", qid, path=db)
        (row,) = list_quizzes("alice", path=db)
        assert row["attempts"] == 2 and row["avg_score"] == pytest.approx(0.75)

    def test_resave_same_id_replaces(self, db):
        qid = save_quiz("alice", [q()], title="v1", path=db)
        save_quiz("alice", [q(), q("two")], title="v2", quiz_id=qid, path=db)
        assert len(list_quizzes("alice", path=db)) == 1 and load_quiz("alice", qid, path=db)["title"] == "v2"

    def test_newest_first_and_limit(self, db):
        ids = [save_quiz("alice", [q()], title=f"t{i}", path=db) for i in range(3)]
        assert [x["id"] for x in list_quizzes("alice", limit=2, path=db)] == ids[::-1][:2]


class TestSpacedRepetition:
    def test_fail_resets_to_tomorrow(self):
        assert next_state(5, 30, 2.5, 0.2) == {"reps": 0, "interval": 1.0, "ease": pytest.approx(2.3)}

    def test_pass_grows_intervals(self):
        s1 = next_state(0, 0, 2.5, 1.0)
        s2 = next_state(s1["reps"], s1["interval"], s1["ease"], 1.0)
        s3 = next_state(s2["reps"], s2["interval"], s2["ease"], 1.0)
        assert (s1["interval"], s2["interval"]) == (1.0, 3.0) and s3["interval"] > 3.0

    def test_ease_is_bounded(self):
        assert next_state(0, 1, 1.3, 0.0)["ease"] == 1.3
        assert next_state(3, 10, 2.8, 1.0)["ease"] == 2.8

    def test_due_after_the_interval_not_before(self, db):
        record_results("alice", [{**q(), "score": 0.0}], now=NOW, path=db)
        assert count_due("alice", now=NOW, path=db) == 0
        assert count_due("alice", now=NOW + timedelta(days=1, minutes=1), path=db) == 1
        (card,) = due_questions("alice", now=NOW + timedelta(days=2), path=db)
        assert card["question"] == "What is X?" and "student" not in card and "score" not in card

    def test_repeated_passes_push_the_date_out(self, db):
        t = NOW
        for _ in range(3):
            record_results("alice", [{**q(), "score": 1.0}], now=t, path=db)
            t += timedelta(days=100)
        assert count_due("alice", now=t - timedelta(days=100) + timedelta(days=2), path=db) == 0

    def test_ungraded_skipped_and_owner_isolation(self, db):
        assert record_results("alice", [{**q(), "score": None, "error": True}], now=NOW, path=db) == 0
        record_results("alice", [{**q(), "score": 0.0}], now=NOW, path=db)
        assert count_due("bob", now=NOW + timedelta(days=5), path=db) == 0

    def test_card_identity(self):
        assert card_id(q()) == card_id(q()) != card_id(q("Other?"))

    def test_limit_and_order(self, db):
        for i in range(3):
            record_results("alice", [{**q(f"Q{i}?"), "score": 0.0}], now=NOW + timedelta(hours=i), path=db)
        got = due_questions("alice", now=NOW + timedelta(days=9), limit=2, path=db)
        assert [c["question"] for c in got] == ["Q0?", "Q1?"]


class TestExports:
    QS = [q("What is\tX?", "X <b>is</b> Y.", explanation="Because."),
          {"question": "Pick one", "answer": "right", "type": "mcq", "options": ["wrong", "right", "no", "nope"],
           "topic": "Big Topic", "source": "n.pdf", "page": 3},
          {"question": "The ___ makes ATP.", "answer": "mitochondria", "type": "cloze"}]

    def test_anki_format(self):
        rows = [r.split("\t") for r in to_anki_tsv(self.QS).strip().split("\n")]
        assert all(len(r) == 3 for r in rows)                      # tabs inside text were removed
        assert "&lt;b&gt;" in rows[0][1] and "<i>Because.</i>" in rows[0][1]
        assert "A) wrong<br>B) right" in rows[1][0] and rows[1][2] == "Big_Topic"
        assert to_anki_tsv([]) == ""

    def test_csv_round_trips(self):
        rows = list(csv.DictReader(io.StringIO(to_csv(self.QS))))
        assert rows[0]["question"] == "What is\tX?" and rows[1]["options"] == "wrong | right | no | nope"
        assert rows[1]["page"] == "3" and rows[2]["type"] == "cloze"

    def test_exam_pdf_with_and_without_key(self):
        plain, keyed = exam_pdf(self.QS, title="Bio"), exam_pdf(self.QS, with_key=True)
        assert plain[:5] == keyed[:5] == b"%PDF-" and len(keyed) > len(plain)

    def test_exam_pdf_handles_odd_text_and_empty(self):
        odd = [{"question": "Ünïcode ✓ " + "x" * 3000, "answer": "ä" * 800, "type": "short"}]
        assert exam_pdf(odd, with_key=True)[:5] == b"%PDF-"
        assert exam_pdf([])[:5] == b"%PDF-"


class TestCleanup:
    def test_expired_docs_removed_from_store_and_registry(self, db, store):
        from app.pdf_loader import Chunk
        store.add_chunks([Chunk("old text here.", 1)], "old", source="a.pdf", file_id="f")
        store.add_chunks([Chunk("new text here.", 1)], "new", source="b.pdf", file_id="g")
        register_doc("old", now=NOW - timedelta(days=30), path=db)
        register_doc("new", now=NOW, path=db)
        assert expired_docs(7, now=NOW, path=db) == ["old"]
        assert cleanup_expired(7, now=NOW, path=db) == 1
        assert store.get_all_chunks("old") == [] and len(store.get_all_chunks("new")) == 1
        assert expired_docs(7, now=NOW, path=db) == []

    def test_zero_ttl_disables(self, db):
        register_doc("old", now=NOW - timedelta(days=999), path=db)
        assert cleanup_expired(0, now=NOW, path=db) == 0 and expired_docs(7, now=NOW, path=db) == ["old"]

    def test_forget_and_failure_tolerance(self, db, monkeypatch):
        register_doc("d", now=NOW - timedelta(days=30), path=db)
        monkeypatch.setattr("app.vector_store.delete_doc", lambda d: (_ for _ in ()).throw(RuntimeError("down")))
        assert cleanup_expired(7, now=NOW, path=db) == 0          # failure is swallowed, doc stays registered
        assert expired_docs(7, now=NOW, path=db) == ["d"]
        forget_doc("d", path=db)
        assert expired_docs(7, now=NOW, path=db) == []


def test_delete_owner_data(db):
    qid = save_quiz("alice", [q()], path=db)
    store_results([{"question": "a", "answer": "b", "score": 1.0}], "alice", qid, path=db)
    record_results("alice", [{**q(), "score": 0.0}], now=NOW, path=db)
    save_quiz("bob", [q()], path=db)
    counts = delete_owner_data("alice", path=db)
    assert counts == {"results": 1, "quizzes": 1, "srs": 1}
    assert list_quizzes("alice", path=db) == [] and len(list_quizzes("bob", path=db)) == 1


# ---------- UI flows ----------
class Up:
    name = "notes.pdf"

    def getvalue(self):
        return make_pdf_bytes([f"Cells divide by mitosis {i}." for i in range(60)])


@pytest.fixture
def ui(store, tmp_path, monkeypatch):
    import streamlit as st
    from app import llm
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "ui.db")

    class Grader(FakeLLM):
        def generate(self, prompt, *, schema=None, **_):
            if "Evaluate the student" in prompt:
                return '{"score": 0.0, "feedback": "Wrong"}'
            return super().generate(prompt, schema=schema)

    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: [Up()])
    monkeypatch.setattr(llm, "get_provider", lambda *a, **k: Grader())
    return AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app" / "ui.py"), default_timeout=60)


def test_quiz_is_saved_and_can_be_retaken(ui):
    ui.run()
    next(t for t in ui.text_input if t.label.startswith("Profile")).set_value("Sam").run()
    next(b for b in ui.button if b.label == "Create Quiz").click().run()
    assert not ui.exception, ui.exception
    first = ui.session_state["qa_pairs"]
    old_attempt = ui.session_state["attempt_id"]
    assert len(list_quizzes("sam")) == 1

    # leave and come back: the saved quiz can be retaken, with fresh answer boxes
    ui.session_state["qa_pairs"] = []
    next(b for b in ui.button if b.label == "Retake").click().run()
    assert not ui.exception, ui.exception
    assert ui.session_state["qa_pairs"] == first and ui.session_state["attempt_id"] != old_attempt


def test_grading_schedules_review_and_review_button_works(ui):
    ui.run()
    next(t for t in ui.text_input if t.label.startswith("Profile")).set_value("Sam").run()
    next(b for b in ui.button if b.label == "Create Quiz").click().run()
    for ta in ui.text_area:
        ta.input("my real attempt")
    next(b for b in ui.button if "grade me" in b.label).click().run()
    assert not ui.exception, ui.exception
    # every answer scored 0 -> due again tomorrow
    assert count_due("sam", now=datetime.now(timezone.utc) + timedelta(days=2)) == 10
    assert count_due("sam") == 0
    assert any("Review due questions (0)" in b.label for b in ui.button)


def test_delete_my_data(ui):
    ui.run()
    next(t for t in ui.text_input if t.label.startswith("Profile")).set_value("Sam").run()
    next(b for b in ui.button if b.label == "Create Quiz").click().run()
    assert list_quizzes("sam")
    next(c for c in ui.checkbox if c.label == "Yes, delete my data").check().run()
    next(b for b in ui.button if b.label == "Delete now").click().run()
    assert not ui.exception, ui.exception
    assert list_quizzes("sam") == []


def test_review_due_button_starts_a_review_quiz(ui):
    past = datetime.now(timezone.utc) - timedelta(days=10)
    record_results("sam", [{**q(f"Old question {i}?"), "score": 0.0} for i in range(3)], now=past)
    ui.run()
    next(t for t in ui.text_input if t.label.startswith("Profile")).set_value("Sam").run()
    review = next(b for b in ui.button if b.label.startswith("🧠 Review due questions"))
    assert "(3)" in review.label and not review.disabled
    review.click().run()
    assert not ui.exception, ui.exception
    assert [x["question"] for x in ui.session_state["qa_pairs"]] == [f"Old question {i}?" for i in range(3)]
    assert "graded" not in ui.session_state
