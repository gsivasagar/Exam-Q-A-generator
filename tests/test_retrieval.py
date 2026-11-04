"""BM25, rank fusion, hybrid retrieval, reranker plumbing, and the 'topic not found' path."""
import pytest

from app import config
from app.bm25 import BM25, tokenize
from app.pdf_loader import Chunk
from app.rerank import FastEmbedReranker, RerankerError, get_reranker
from app.retrieval import retrieve, rrf_fuse


class TestBM25:
    def test_tokenize_drops_stopwords_and_plurals(self):
        assert tokenize("The cells ARE dividing") == ["cell", "dividing"]
        assert tokenize("class") == ["class"]            # "ss" is not a plural
        assert "o(n-log-n)" not in tokenize("x")         # no crash on odd input

    def test_exact_term_ranks_first(self):
        docs = ["General remarks about algorithms and data.",
                "The TCP three-way handshake uses SYN and ACK packets.",
                "Photosynthesis happens in chloroplasts."]
        assert BM25(docs).rank("SYN ACK", k=3)[0] == 1

    def test_no_match_returns_nothing(self):
        assert BM25(["alpha beta", "gamma delta"]).rank("zebra", k=5) == []

    def test_empty_corpus(self):
        assert BM25([]).rank("x", k=3) == []

    def test_rarer_terms_weigh_more(self):
        docs = ["common common rare", "common common common", "common other"]
        assert BM25(docs).rank("common rare", k=3)[0] == 0


class TestRRF:
    def test_item_ranked_well_by_both_wins(self):
        assert rrf_fuse([[1, 2, 3], [3, 1, 2]])[0] == 1

    def test_union_of_items_and_deterministic_ties(self):
        assert sorted(rrf_fuse([[1, 2], [3, 4]])) == [1, 2, 3, 4]
        assert rrf_fuse([[1], [2]]) == rrf_fuse([[1], [2]])

    def test_empty(self):
        assert rrf_fuse([]) == [] and rrf_fuse([[], []]) == []


@pytest.fixture
def notes(store):
    texts = [
        "The mitochondria is the organelle that produces ATP through cellular respiration.",
        "Photosynthesis converts light energy into chemical energy inside chloroplasts.",
        "The TCP handshake uses SYN, SYN-ACK and ACK segments to open a connection.",
        "Binary search runs in O(log n) time on a sorted array.",
        "The French Revolution began in 1789 with the storming of the Bastille.",
    ]
    store.add_chunks([Chunk(t, i + 1) for i, t in enumerate(texts)], "d", source="n.pdf", file_id="f")
    return store


class TestRetrieve:
    def test_dense_mode_matches_search(self, notes):
        hits = retrieve("cellular energy production", "d", k=2, mode="dense")
        assert len(hits) == 2 and any("mitochondria" in h["text"] for h in hits)

    def test_hybrid_finds_exact_term(self, notes):
        hits = retrieve("SYN-ACK", "d", k=2, mode="hybrid")
        assert "TCP" in hits[0]["text"]

    def test_hybrid_respects_k_and_returns_best_first(self, notes):
        assert len(retrieve("energy", "d", k=3, mode="hybrid")) == 3

    def test_hybrid_absent_topic_returns_nothing_with_cutoff(self, notes, monkeypatch):
        monkeypatch.setattr(config, "RETRIEVAL_MAX_DISTANCE", 0.2)
        assert retrieve("quantum chromodynamics gluons", "d", k=3, mode="hybrid") == []

    def test_unknown_document_set(self, notes):
        assert retrieve("anything", "missing", k=3, mode="hybrid") == []

    def test_bad_mode(self, notes):
        with pytest.raises(ValueError):
            retrieve("x", "d", mode="magic")

    def test_distance_cutoff_filters_dense(self, notes, monkeypatch):
        monkeypatch.setattr(config, "RETRIEVAL_MAX_DISTANCE", 0.0001)
        assert retrieve("cellular energy production", "d", k=5, mode="dense") == []

    def test_reranker_reorders(self, notes):
        class Reverse:
            def rerank(self, query, hits):
                return list(reversed(hits))

        base = retrieve("energy", "d", k=3, mode="dense")
        out = retrieve("energy", "d", k=3, mode="dense", reranker=Reverse(), candidates=3)
        assert [h["text"] for h in out] == [h["text"] for h in reversed(base)]


class TestReranker:
    def test_none_by_default(self):
        assert get_reranker("none") is None and get_reranker("") is None

    def test_bad_spec(self):
        with pytest.raises(ValueError):
            get_reranker("openai:x")

    def test_missing_fastembed_gives_actionable_error(self, monkeypatch):
        import builtins
        real = builtins.__import__

        def fake(name, *a, **k):
            if name.startswith("fastembed"):
                raise ImportError("nope")
            return real(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", fake)
        with pytest.raises(RerankerError, match="pip install fastembed"):
            FastEmbedReranker("some/model")

    def test_scores_order_hits(self):
        class Enc:
            def rerank(self, query, docs):
                return [len(d) for d in docs]   # longest first

        r = FastEmbedReranker("m", encoder=Enc())
        out = r.rerank("q", [{"text": "aa"}, {"text": "a"}, {"text": "aaa"}])
        assert [h["text"] for h in out] == ["aaa", "aa", "a"]

    def test_failure_wrapped(self):
        class Bad:
            def rerank(self, query, docs):
                raise RuntimeError("boom")

        with pytest.raises(RerankerError):
            FastEmbedReranker("m", encoder=Bad()).rerank("q", [{"text": "a"}, {"text": "b"}])

    def test_single_hit_not_sent_to_model(self):
        class Bad:
            def rerank(self, *a):
                raise AssertionError("should not be called")

        assert FastEmbedReranker("m", encoder=Bad()).rerank("q", [{"text": "a"}]) == [{"text": "a"}]


def test_absent_topic_gives_clear_error(notes, monkeypatch):
    from app.qa_generator import generate_qa_pairs
    from tests.conftest import FakeLLM

    monkeypatch.setattr(config, "RETRIEVAL_MODE", "hybrid")
    monkeypatch.setattr(config, "RETRIEVAL_MAX_DISTANCE", 0.2)
    with pytest.raises(ValueError, match="not found in your notes"):
        generate_qa_pairs("d", FakeLLM(), n=5, topic="quantum chromodynamics gluons")


class TestAutoMode:
    """RETRIEVAL_MODE=auto: hybrid for documents of at least HYBRID_MIN_CHUNKS chunks, dense below."""

    @pytest.fixture
    def spy(self, monkeypatch):
        """Records whether the hybrid path (the only caller of get_all_chunks here) ran."""
        from app import retrieval

        calls = []
        real = retrieval.get_all_chunks
        monkeypatch.setattr(retrieval, "get_all_chunks", lambda d: (calls.append(d), real(d))[1])
        return calls

    def test_default_mode_is_auto(self):
        import os
        import subprocess
        import sys

        env = {k: v for k, v in os.environ.items() if k not in ("RETRIEVAL_MODE", "HYBRID_MIN_CHUNKS")}
        out = subprocess.run([sys.executable, "-c", "from app import config; "
                              "print(config.RETRIEVAL_MODE, config.HYBRID_MIN_CHUNKS)"],
                             capture_output=True, text=True, env=env, check=True).stdout.split()
        assert out == ["auto", "20"]

    def test_count_chunks(self, notes):
        from app.vector_store import count_chunks

        assert count_chunks("d") == 5 and count_chunks("missing") == 0

    def test_long_document_uses_hybrid(self, notes, monkeypatch, spy):
        monkeypatch.setattr(config, "RETRIEVAL_MODE", "auto")
        monkeypatch.setattr(config, "HYBRID_MIN_CHUNKS", 5)          # the notes have exactly 5 chunks
        assert "TCP" in retrieve("SYN-ACK", "d", k=2)[0]["text"]
        assert spy == ["d"]

    def test_short_document_uses_dense(self, notes, monkeypatch, spy):
        monkeypatch.setattr(config, "RETRIEVAL_MODE", "auto")
        monkeypatch.setattr(config, "HYBRID_MIN_CHUNKS", 6)
        retrieve("cellular energy production", "d", k=2)
        assert spy == []

    def test_explicit_modes_ignore_the_size_rule(self, notes, monkeypatch, spy):
        monkeypatch.setattr(config, "HYBRID_MIN_CHUNKS", 1)
        retrieve("energy", "d", k=2, mode="dense")
        assert spy == []                                              # dense stays dense on a "long" document
        monkeypatch.setattr(config, "HYBRID_MIN_CHUNKS", 10_000)
        retrieve("energy", "d", k=2, mode="hybrid")
        assert spy == ["d"]                                           # hybrid stays hybrid on a "short" one

    def test_config_mode_dense_is_still_available(self, notes, monkeypatch, spy):
        monkeypatch.setattr(config, "RETRIEVAL_MODE", "dense")
        monkeypatch.setattr(config, "HYBRID_MIN_CHUNKS", 1)
        retrieve("energy", "d", k=2)
        assert spy == []

    def test_unknown_document_set_is_empty(self, notes, monkeypatch):
        monkeypatch.setattr(config, "RETRIEVAL_MODE", "auto")
        assert retrieve("anything", "missing", k=3) == []
