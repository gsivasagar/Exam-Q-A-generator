import pytest

from app.qa_generator import (extract_json_array, generate_qa_pairs, ingest_pdf,
                              validate_pairs)
from tests.conftest import FakeLLM


class TestExtractJson:
    def test_plain(self):
        assert extract_json_array('[{"question":"q","answer":"a"}]')[0]["question"] == "q"

    def test_prose_and_fences(self):
        t = 'Sure! Here you go:\n```json\n[{"question":"q","answer":"a"}]\n```\nHope it helps'
        assert len(extract_json_array(t)) == 1

    def test_regression_brackets_inside_strings(self):
        # the old non-greedy regex cut the text at the "}]" inside the first answer
        t = ('[{"question":"What does a list of dicts look like?",'
             '"answer":"e.g. [{\\"a\\": 1}] is one","topic":"T"},'
             '{"question":"q2","answer":"y","topic":"T"}]')
        assert len(extract_json_array(t)) == 2

    def test_trailing_comma_and_curly_quotes(self):
        t = '[{“question”:"q","answer":"a",},]'
        assert len(extract_json_array(t)) == 1

    def test_object_wrapping_list(self):
        assert len(extract_json_array('{"questions":[{"question":"q","answer":"a"}]}')) == 1

    def test_single_bare_question_object(self):
        # what Ollama's format="json" + gemma3 actually returned in a live run
        assert extract_json_array('{"question":"q","answer":"a","topic":"T"}')[0]["answer"] == "a"

    def test_garbage(self):
        with pytest.raises(ValueError):
            extract_json_array("I cannot do that.")


class TestValidate:
    def test_drops_malformed_and_defaults_topic(self):
        items = [{"question": "q", "answer": "a"}, {"question": "no answer"}, "str", {"answer": "x"},
                 {"question": " ", "answer": "a"}, {"question": "q2", "answer": "a2", "topic": "T"}]
        out = validate_pairs(items)
        assert [o["question"] for o in out] == ["q", "q2"]
        assert out[0]["topic"] == "General" and out[1]["topic"] == "T"


class TestGenerate:
    def test_multi_batch_returns_exactly_n(self, store, pdf_bytes):
        ingest_pdf(None, "d1", filename="a.pdf", data=pdf_bytes)
        llm = FakeLLM()
        pairs = generate_qa_pairs("d1", llm, n=12)
        assert len(pairs) == 12 and llm.calls >= 3
        assert len({p["question"] for p in pairs}) == 12

    def test_retries_on_garbage_then_succeeds(self, store, pdf_bytes):
        ingest_pdf(None, "d2", filename="a.pdf", data=pdf_bytes)
        llm = FakeLLM(["not json at all", '[{"question":"q?","answer":"a","topic":"T"}]'])
        assert generate_qa_pairs("d2", llm, n=1)[0]["question"] == "q?"

    def test_all_garbage_raises_friendly_error(self, store, pdf_bytes):
        ingest_pdf(None, "d3", filename="a.pdf", data=pdf_bytes)
        with pytest.raises(ValueError, match="usable questions"):
            generate_qa_pairs("d3", FakeLLM(["nope"] * 50), n=5)

    def test_unknown_doc_raises(self, store):
        with pytest.raises(ValueError, match="No content"):
            generate_qa_pairs("missing", FakeLLM(), n=5)

    def test_topic_filter_uses_topic_in_prompt(self, store, pdf_bytes):
        ingest_pdf(None, "d4", filename="a.pdf", data=pdf_bytes)
        llm = FakeLLM()
        generate_qa_pairs("d4", llm, n=3, topic="mitochondria")
        assert "mitochondria" in llm.prompts[0]
