"""Choosing which document chunks go into each LLM call."""
from __future__ import annotations

import random
from typing import List, Sequence, Union

from .vector_store import get_all_chunks, similarity_search

CHUNKS_PER_BATCH = 4  # distinct context chunks per call
TOPIC_TOP_K = 30      # chunks fetched per topic


def normalize_topics(topic: Union[None, str, Sequence[str]]) -> List[str]:
    """Accept one topic string or a list; drop blanks and non-strings."""
    if isinstance(topic, str):
        return [topic] if topic.strip() else []
    return [t for t in (topic or []) if isinstance(t, str) and t.strip()]


def gather(doc_id: str, topics: List[str]) -> List[dict]:
    """All chunks of the document, or (with topics) the de-duplicated union of per-topic searches."""
    if not topics:
        return get_all_chunks(doc_id)
    seen, hits = set(), []
    for t in topics:
        for h in similarity_search(t, k=TOPIC_TOP_K, doc_id=doc_id):
            if h["text"] not in seen:
                seen.add(h["text"])
                hits.append(h)
    return hits


class BatchCycler:
    """Shuffles chunks once, then hands out consecutive batches, wrapping around."""

    def __init__(self, hits: List[dict]):
        self.hits = hits
        random.shuffle(self.hits)
        self._pos = 0

    def next_batch(self) -> List[dict]:
        size = min(CHUNKS_PER_BATCH, len(self.hits))
        batch = [self.hits[(self._pos + i) % len(self.hits)] for i in range(size)]
        self._pos += CHUNKS_PER_BATCH
        return batch
