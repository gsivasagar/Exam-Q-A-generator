import pytest

from app.grader import grade, parse_grade
from app.llm import LLMError
from tests.conftest import FakeLLM


@pytest.mark.parametrize("text,score", [
    ('{"score": 0.8, "feedback": "Good"}', 0.8),
    ('Here: {"score": "0.5", "feedback": "ok"}', 0.5),
    ("Score: 8/10\nFeedback: nice", 0.8),
    ("Score: 0.8/1.0\nFeedback: nice", 0.8),
    ('{"score": 7, "feedback": "x"}', 0.7),          # 0-10 scale
    ('{"score": 85, "feedback": "x"}', 0.85),        # 0-100 scale
    ('{"score": "75%", "feedback": "x"}', 0.75),
    ('{"score": 3.0, "feedback": "x"}', 0.3),
    ('{"score": -2, "feedback": "x"}', 0.0),         # clamped
])
def test_parse_scores(text, score):
    assert parse_grade(text)["score"] == pytest.approx(score)


def test_parse_feedback_text_fallback():
    assert parse_grade("Score: 8/10\nFeedback: nice work")["feedback"] == "nice work"


def test_unparseable_raises():
    with pytest.raises(ValueError):
        parse_grade("I think it's good")


def test_grade_failure_is_not_zero():
    r = grade("ref", "stu", FakeLLM(["garbage"]))
    assert r["score"] is None and r["error"]


def test_grade_llm_error_is_not_zero():
    r = grade("ref", "stu", FakeLLM([LLMError("down")]))
    assert r["score"] is None and r["error"]


def test_grade_ok():
    assert grade("ref", "stu", FakeLLM(['{"score":1,"feedback":"perfect"}']))["score"] == 1.0
