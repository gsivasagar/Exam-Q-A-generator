"""Adaptive routing: path choice, budgets, and how generate_qa_pairs uses each path."""
import pytest

from app import config
from app.qa_generator import generate_qa_pairs, ingest_pdf
from app.router import COVERAGE, FULL, TOPIC, choose_path, context_budget, document_chars
from tests.conftest import FakeLLM, make_pdf_bytes


def hits(n, size=100):
    return [{"text": "x" * size} for _ in range(n)]


class TestChoosePath:
    def test_small_document_goes_full(self):
        assert choose_path(hits(3), [], budget=6000) == FULL
        assert choose_path(hits(3), ["topic"], budget=6000) == FULL

    def test_large_document_routes_by_topic(self):
        assert choose_path(hits(100), [], budget=6000) == COVERAGE
        assert choose_path(hits(100), ["t"], budget=6000) == TOPIC

    def test_disabled_never_full(self):
        assert choose_path(hits(1), [], budget=6000, enabled=False) == COVERAGE

    def test_boundary_counts_prompt_overhead(self):
        h = hits(10, 100)
        exact = document_chars(h)
        assert choose_path(h, [], budget=exact) == FULL
        assert choose_path(h, [], budget=exact - 1) == COVERAGE


class TestBudget:
    def test_default_and_declared(self):
        class P:
            context_chars = 50000
        assert context_budget(None) == config.FULL_CONTEXT_MAX_CHARS
        assert context_budget(FakeLLM()) == config.FULL_CONTEXT_MAX_CHARS
        assert context_budget(P()) == 50000

    def test_providers_declare_budgets(self, monkeypatch):
        from app.llm import GeminiProvider, OllamaProvider
        assert OllamaProvider(model="m", host="http://x:1", num_ctx=8192).context_chars == 16384
        assert OllamaProvider(model="m", host="http://x:1").context_chars is None
        assert GeminiProvider("k").context_chars == config.GEMINI_CONTEXT_CHARS


def long_notes():
    return make_pdf_bytes([f"Topic {i // 8} detail sentence number {i} explains something."
                           for i in range(900)])


def short_notes():
    return make_pdf_bytes([f"Mitochondria produce energy number {i} for the cell." for i in range(20)])


class TestRouting:
    def test_short_doc_sends_whole_document_every_call(self, store):
        ingest_pdf(None, "s", filename="a.pdf", data=short_notes())
        seen = []
        out = generate_qa_pairs("s", FakeLLM(), n=5, on_route=seen.append)
        assert seen == [FULL] and len(out) == 5

    def test_short_doc_prompt_contains_all_text(self, store):
        ingest_pdf(None, "s", filename="a.pdf", data=short_notes())
        llm = FakeLLM()
        generate_qa_pairs("s", llm, n=5)
        assert "number 0" in llm.prompts[0] and "number 19" in llm.prompts[0]

    def test_long_doc_uses_coverage(self, store):
        ingest_pdf(None, "l", filename="a.pdf", data=long_notes())
        seen = []
        generate_qa_pairs("l", FakeLLM(), n=5, on_route=seen.append)
        assert seen == [COVERAGE]

    def test_long_doc_with_topic_uses_retrieval(self, store):
        ingest_pdf(None, "l", filename="a.pdf", data=long_notes())
        seen = []
        generate_qa_pairs("l", FakeLLM(), n=5, topic="Topic 3", on_route=seen.append)
        assert seen == [TOPIC]

    def test_routing_can_be_disabled(self, store, monkeypatch):
        monkeypatch.setattr(config, "ADAPTIVE_ROUTING", False)
        ingest_pdf(None, "s", filename="a.pdf", data=short_notes())
        seen = []
        generate_qa_pairs("s", FakeLLM(), n=5, on_route=seen.append)
        assert seen == [COVERAGE]

    def test_full_context_batch_reduces_calls(self, store, monkeypatch):
        ingest_pdf(None, "s", filename="a.pdf", data=short_notes())
        monkeypatch.setattr(config, "FULL_CONTEXT_BATCH", 5)
        a = FakeLLM()
        generate_qa_pairs("s", a, n=10)
        monkeypatch.setattr(config, "FULL_CONTEXT_BATCH", 12)
        b = FakeLLM()
        generate_qa_pairs("s", b, n=10)
        assert b.calls <= a.calls

    def test_absent_topic_still_reported_on_short_docs(self, store, monkeypatch):
        monkeypatch.setattr(config, "RETRIEVAL_MODE", "hybrid")
        monkeypatch.setattr(config, "RETRIEVAL_MAX_DISTANCE", 0.2)
        ingest_pdf(None, "s", filename="a.pdf", data=short_notes())
        with pytest.raises(ValueError, match="not found in your notes"):
            generate_qa_pairs("s", FakeLLM(), n=5, topic="quantum chromodynamics gluons")

    def test_no_content_still_errors(self, store):
        with pytest.raises(ValueError, match="No content found"):
            generate_qa_pairs("missing", FakeLLM(), n=5)
