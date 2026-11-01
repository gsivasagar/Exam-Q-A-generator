"""Adaptive routing: pick how context is chosen for a quiz.

  full      The whole document fits the model's context budget: skip selection and send everything.
            No retrieval error is possible and every part of the notes is available.
  topic     Longer document with a topic filter: retrieve the relevant chunks (dense / hybrid).
  coverage  Longer document, no topic: pick chunks that span the whole document.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from . import config

FULL, TOPIC, COVERAGE = "full", "topic", "coverage"
_PER_PASSAGE_OVERHEAD = 8  # "[12] " marker plus separators in the prompt


def document_chars(hits: Sequence[dict]) -> int:
    """Characters the whole document would occupy in a prompt."""
    return sum(len(h["text"]) + _PER_PASSAGE_OVERHEAD for h in hits)


def context_budget(llm=None) -> int:
    """Characters of context we are willing to send in one call: the provider's own budget when it
    declares one (`context_chars`), else FULL_CONTEXT_MAX_CHARS."""
    declared = getattr(llm, "context_chars", None)
    return int(declared) if declared else config.FULL_CONTEXT_MAX_CHARS


def choose_path(all_hits: Sequence[dict], topics: List[str], budget: int,
                enabled: Optional[bool] = None) -> str:
    enabled = config.ADAPTIVE_ROUTING if enabled is None else enabled
    if enabled and document_chars(all_hits) <= budget:
        return FULL
    return TOPIC if topics else COVERAGE
