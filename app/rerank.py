"""Optional second-stage reranking of retrieved chunks.

RERANKER is "none" (default) or "fastembed:<model>", e.g. "fastembed:BAAI/bge-reranker-base". The
fastembed backend is optional (`pip install fastembed`) and downloads the model on first use.
"""
from __future__ import annotations

from typing import List, Optional, Protocol

from . import config
from .llm import LLMError


class Reranker(Protocol):
    def rerank(self, query: str, hits: List[dict]) -> List[dict]: ...


class RerankerError(LLMError):
    """The reranker backend is missing or failed."""


class FastEmbedReranker:
    def __init__(self, model: str, encoder=None):
        self.model = model
        if encoder is None:
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder
            except ImportError as exc:
                raise RerankerError(
                    "RERANKER is set to a fastembed model but fastembed is not installed. "
                    "Run `pip install fastembed`, or set RERANKER=none.") from exc
            try:
                encoder = TextCrossEncoder(model_name=model)
            except Exception as exc:
                raise RerankerError(f"Could not load reranker model '{model}': {exc}") from exc
        self._encoder = encoder

    def rerank(self, query: str, hits: List[dict]) -> List[dict]:
        if len(hits) < 2:
            return list(hits)
        try:
            scores = list(self._encoder.rerank(query, [h["text"] for h in hits]))
        except Exception as exc:
            raise RerankerError(f"Reranking failed: {exc}") from exc
        order = sorted(range(len(hits)), key=lambda i: -float(scores[i]))
        return [hits[i] for i in order]


def get_reranker(spec: Optional[str] = None) -> Optional[Reranker]:
    spec = (spec if spec is not None else config.RERANKER).strip()
    if not spec or spec.lower() == "none":
        return None
    kind, _, model = spec.partition(":")
    if kind.lower() == "fastembed" and model.strip():
        return FastEmbedReranker(model.strip())
    raise ValueError(f"RERANKER must be 'none' or 'fastembed:<model>', got {spec!r}")
