"""Embedder selection: spec parsing, per-embedder collections, Ollama embedder with a fake client."""
import pytest

from app import config, vector_store
from app.embeddings import (EmbeddingError, OllamaEmbedder, collection_name, get_embedding_function,
                            parse_spec)
from app.llm import LLMError
from app.pdf_loader import Chunk


class TestSpec:
    def test_parse(self):
        assert parse_spec(None) == ("default", "")
        assert parse_spec(" DEFAULT ") == ("default", "")
        assert parse_spec("ollama:bge-m3") == ("ollama", "bge-m3")
        assert parse_spec("ollama:qwen3-embedding:0.6b") == ("ollama", "qwen3-embedding:0.6b")

    def test_fastembed_spec(self):
        assert parse_spec("fastembed:BAAI/bge-small-en-v1.5") == ("fastembed", "BAAI/bge-small-en-v1.5")
        assert collection_name("fastembed:BAAI/bge-small-en-v1.5") == "exam_chunks__fastembed_BAAI_bge-small-en-v1.5"

    @pytest.mark.parametrize("bad", ["openai:x", "ollama:", "bge-m3", "ollama", "fastembed:"])
    def test_rejects_bad_specs(self, bad):
        with pytest.raises(ValueError):
            parse_spec(bad)

    def test_collection_names(self):
        assert collection_name("default") == "exam_chunks"  # unchanged name = existing stores keep working
        assert collection_name("ollama:bge-m3") == "exam_chunks__ollama_bge-m3"
        assert collection_name("ollama:qwen3-embedding:0.6b") == "exam_chunks__ollama_qwen3-embedding_0.6b"
        assert collection_name("ollama:a") != collection_name("ollama:b")


class FakeClient:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls = result, error, []

    def embed(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return {"embeddings": self.result}


class TestOllamaEmbedder:
    def test_embeds_batch(self):
        c = FakeClient([[1, 0], [0, 1]])
        out = OllamaEmbedder("m", client=c)(["a", "b"])
        assert out == [[1.0, 0.0], [0.0, 1.0]]
        assert c.calls == [{"model": "m", "input": ["a", "b"]}]

    def test_empty_input_makes_no_call(self):
        c = FakeClient([])
        assert OllamaEmbedder("m", client=c)([]) == []
        assert c.calls == []

    def test_backend_failure_is_friendly_and_an_llmerror(self):
        with pytest.raises(EmbeddingError, match="ollama pull m") as ei:
            OllamaEmbedder("m", client=FakeClient(error=ConnectionError("down")))(["x"])
        assert isinstance(ei.value, LLMError)  # existing UI error handling catches it

    def test_count_mismatch_rejected(self):
        with pytest.raises(EmbeddingError):
            OllamaEmbedder("m", client=FakeClient([[1.0]]))(["a", "b"])

    def test_chroma_protocol_bits(self):
        e = OllamaEmbedder("m", client=FakeClient([]))
        assert e.name() == "exam_ollama_embedder" and e.get_config() == {"model": "m"}
        assert e.default_space() == "cosine" and "cosine" in e.supported_spaces()


def test_factory_returns_ollama_embedder(monkeypatch):
    monkeypatch.setattr("ollama.Client", lambda **kw: FakeClient([]))
    assert isinstance(get_embedding_function("ollama:bge-m3"), OllamaEmbedder)


class TestStoreUsesConfiguredEmbedder:
    def test_separate_collections_per_embedder(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "CHROMA_DIR", str(tmp_path / "c"))
        vector_store._collection.cache_clear()
        monkeypatch.setattr(config, "EMBEDDING_MODEL", "default")
        default_col = vector_store._collection()
        assert default_col.name == "exam_chunks"

        class Tiny:
            """3-d fake embedder: proves a different vector size can coexist."""
            def __call__(self, input):  # noqa: A002
                return [[float(len(t) % 3), 1.0, 0.5] for t in input]

            @staticmethod
            def name():
                return "tiny"

            def get_config(self):
                return {}

            def is_legacy(self):
                return False

        monkeypatch.setattr("app.vector_store.get_embedding_function", lambda spec: Tiny())
        monkeypatch.setattr(config, "EMBEDDING_MODEL", "ollama:tiny")
        tiny_col = vector_store._collection()
        assert tiny_col.name == "exam_chunks__ollama_tiny" and tiny_col is not default_col

        vector_store.add_chunks([Chunk("hello world.", 1)], "d", source="s.pdf", file_id="f")
        assert vector_store.get_all_chunks("d")[0]["text"] == "hello world."
        assert len(vector_store.get_all_chunks("d")[0]["embedding"]) == 3
        # the default-embedder collection is untouched
        monkeypatch.setattr(config, "EMBEDDING_MODEL", "default")
        assert vector_store.get_all_chunks("d") == []
        vector_store._collection.cache_clear()

    def test_changing_chroma_dir_is_respected(self, tmp_path, monkeypatch):
        vector_store._collection.cache_clear()
        monkeypatch.setattr(config, "CHROMA_DIR", str(tmp_path / "a"))
        a = vector_store._collection()
        monkeypatch.setattr(config, "CHROMA_DIR", str(tmp_path / "b"))
        assert vector_store._collection() is not a
        vector_store._collection.cache_clear()


class TestFastEmbedEmbedder:
    class Enc:
        def embed(self, texts):
            return ([float(len(t)), 1.0] for t in texts)

    def test_embeds(self):
        from app.embeddings import FastEmbedEmbedder
        assert FastEmbedEmbedder("m", encoder=self.Enc())(["ab", "abc"]) == [[2.0, 1.0], [3.0, 1.0]]
        assert FastEmbedEmbedder("m", encoder=self.Enc())([]) == []

    def test_missing_package_message(self, monkeypatch):
        import builtins
        real = builtins.__import__

        def fake(name, *a, **k):
            if name == "fastembed":
                raise ImportError("nope")
            return real(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", fake)
        from app.embeddings import FastEmbedEmbedder
        with pytest.raises(EmbeddingError, match="pip install fastembed"):
            FastEmbedEmbedder("m")

    def test_failure_wrapped_and_protocol(self):
        from app.embeddings import FastEmbedEmbedder

        class Bad:
            def embed(self, texts):
                raise RuntimeError("boom")

        with pytest.raises(EmbeddingError):
            FastEmbedEmbedder("m", encoder=Bad())(["x"])
        e = FastEmbedEmbedder("m", encoder=self.Enc())
        assert e.name() == "exam_fastembed_embedder" and e.default_space() == "cosine"


class TestSearchThroughRealChroma:
    """Regression: Chroma calls embed_query when searching; custom embedders without it crashed every search."""

    class Client:
        def embed(self, model, input):  # noqa: A002
            return {"embeddings": [[float(len(t) % 5), float(t.count("a")), 1.0] for t in input]}

    def _use(self, store, monkeypatch, make):
        from app import config as cfg
        monkeypatch.setattr(cfg, "EMBEDDING_MODEL", "ollama:fake")
        monkeypatch.setattr("app.vector_store.get_embedding_function", lambda spec: make())
        store._collection.cache_clear()

    def test_ollama_embedder_supports_search(self, store, monkeypatch):
        self._use(store, monkeypatch, lambda: OllamaEmbedder("fake", client=self.Client()))
        store.add_chunks([Chunk("banana bread recipe.", 1), Chunk("tcp handshake notes.", 1)], "d",
                         source="s.pdf", file_id="f")
        hits = store.similarity_search("banana", k=2, doc_id="d")
        assert len(hits) == 2 and all("distance" in h for h in hits)

    def test_fastembed_embedder_supports_search(self, store, monkeypatch):
        from app.embeddings import FastEmbedEmbedder

        class Enc:
            def embed(self, texts):
                return ([float(len(t) % 5), float(t.count("a")), 1.0] for t in texts)

        self._use(store, monkeypatch, lambda: FastEmbedEmbedder("fake", encoder=Enc()))
        store.add_chunks([Chunk("banana bread recipe.", 1), Chunk("tcp handshake notes.", 1)], "d",
                         source="s.pdf", file_id="f")
        assert len(store.similarity_search("banana", k=2, doc_id="d")) == 2
