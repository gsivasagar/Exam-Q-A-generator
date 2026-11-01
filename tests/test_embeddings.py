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

    @pytest.mark.parametrize("bad", ["openai:x", "ollama:", "bge-m3", "ollama"])
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
