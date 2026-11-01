"""Embedding-model selection for the vector store.

`EMBEDDING_MODEL` is one of:
  - "default"        Chroma's built-in ONNX all-MiniLM-L6-v2 (no setup, English-centric)
  - "ollama:<model>" any embedding model served by Ollama, e.g. "ollama:bge-m3",
                     "ollama:nomic-embed-text", "ollama:qwen3-embedding:0.6b"

Each embedder gets its own Chroma collection, because vectors from different models have different
sizes and meanings and must never be mixed.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from . import config
from .llm import LLMError

DEFAULT_SPEC = "default"


class EmbeddingError(LLMError):
    """The embedding backend could not be reached or returned something unusable."""


def parse_spec(spec: Optional[str]) -> Tuple[str, str]:
    """("default", "") or ("ollama", "<model>"). Raises ValueError on anything else."""
    spec = (spec or DEFAULT_SPEC).strip()
    if spec.lower() == DEFAULT_SPEC:
        return DEFAULT_SPEC, ""
    kind, _, model = spec.partition(":")
    if kind.lower() == "ollama" and model.strip():
        return "ollama", model.strip()
    raise ValueError(f"EMBEDDING_MODEL must be 'default' or 'ollama:<model>', got {spec!r}")


def collection_name(spec: Optional[str]) -> str:
    """Stable Chroma collection name per embedder ('exam_chunks' for the default, for compatibility)."""
    kind, model = parse_spec(spec)
    if kind == DEFAULT_SPEC:
        return "exam_chunks"
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("_.-")[:60] or "model"
    return f"exam_chunks__ollama_{slug}"


class OllamaEmbedder:
    """Chroma-compatible embedding function backed by an Ollama server."""

    def __init__(self, model: str, host: Optional[str] = None, timeout: Optional[float] = None, client=None):
        self.model = model
        if client is None:
            from ollama import Client
            client = Client(host=host or config.OLLAMA_URL,
                            timeout=timeout if timeout is not None else config.LLM_TIMEOUT)
        self._client = client

    @staticmethod
    def name() -> str:
        return "exam_ollama_embedder"

    def get_config(self) -> dict:
        return {"model": self.model}

    def is_legacy(self) -> bool:
        return False

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> List[str]:
        return ["cosine", "l2", "ip"]

    def __call__(self, input: List[str]) -> List[List[float]]:  # noqa: A002 - Chroma's parameter name
        texts = list(input)
        if not texts:
            return []
        try:
            resp = self._client.embed(model=self.model, input=texts)
            vectors = resp["embeddings"]
        except Exception as exc:
            raise EmbeddingError(
                f"Ollama embedding request failed ({exc}). Is Ollama running and is "
                f"'{self.model}' pulled? (ollama pull {self.model})"
            ) from exc
        if len(vectors) != len(texts):
            raise EmbeddingError(f"Ollama returned {len(vectors)} embeddings for {len(texts)} texts.")
        return [list(map(float, v)) for v in vectors]


def get_embedding_function(spec: Optional[str] = None):
    """Build the Chroma embedding function for `spec` (default: config.EMBEDDING_MODEL)."""
    kind, model = parse_spec(spec if spec is not None else config.EMBEDDING_MODEL)
    if kind == "ollama":
        return OllamaEmbedder(model)
    # Chroma's built-in ONNX build of all-MiniLM-L6-v2: no torch / sentence-transformers dependency.
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    return DefaultEmbeddingFunction()
