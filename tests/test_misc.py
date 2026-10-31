from app.database import load_results, store_results
from app.recommendation import recommend
from app.report import build_pdf


def _g(topic, score):
    return {"question": f"q-{topic}", "answer": "a", "topic": topic, "student": "s",
            "score": score, "feedback": "f"}


def test_recommend_ignores_ungraded_and_ranks():
    graded = [_g("A", 0.1), _g("A", 0.2), _g("B", 0.5), _g("C", 0.9), _g("D", None)]
    assert recommend(graded) == ["A", "B"]


def test_recommend_missing_topic_does_not_crash():
    assert recommend([{"score": 0.0}]) == ["General"]


def test_db_scoped_by_session_and_skips_ungraded(tmp_path):
    p = tmp_path / "r.db"
    store_results([_g("A", 0.5), _g("B", None)], "s1", "q1", path=p)
    store_results([_g("C", 0.9)], "s2", "q2", path=p)
    s1 = load_results("s1", path=p)
    assert list(s1["topic"]) == ["A"]
    assert len(load_results(path=p)) == 2


def test_pdf_report_handles_odd_text():
    graded = [_g("A", 0.5), _g("B", None)]
    graded[0]["question"] = "x" * 500 + " ✓ 中文 emoji \U0001F600"
    out = build_pdf(graded)
    assert out[:4] == b"%PDF"
