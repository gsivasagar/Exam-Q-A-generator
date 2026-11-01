"""Choosing which document chunks go into each LLM call."""
from __future__ import annotations

import random
from collections import deque
from typing import Deque, List, Optional, Sequence, Union

import numpy as np

from . import config
from .clustering import farthest_first_order, kmeans
from .retrieval import retrieve
from .vector_store import get_all_chunks

CHUNKS_PER_BATCH = 4  # distinct context chunks per call
TOPIC_TOP_K = 30      # chunks fetched per topic
MAX_CLUSTERS = 12     # upper bound on k for coverage selection
STRATEGIES = ("coverage", "random")


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
        for h in retrieve(t, doc_id, k=TOPIC_TOP_K):
            if h["text"] not in seen:
                seen.add(h["text"])
                hits.append(h)
    return hits


class BatchCycler:
    """Random strategy: shuffle once, then hand out consecutive batches, wrapping around."""

    def __init__(self, hits: List[dict]):
        self.hits = hits
        random.shuffle(self.hits)
        self._pos = 0

    def next_batch(self) -> List[dict]:
        size = min(CHUNKS_PER_BATCH, len(self.hits))
        batch = [self.hits[(self._pos + i) % len(self.hits)] for i in range(size)]
        self._pos += CHUNKS_PER_BATCH
        return batch


class CoverageCycler:
    """Coverage strategy: group chunks by embedding similarity, then build each batch by taking one
    chunk from each of several different groups, rotating through the groups across batches.

    Result: every batch mixes different parts of the document, and over a quiz all parts get visited
    before any part is revisited. Needs an `embedding` on every hit (see `make_cycler`)."""

    def __init__(self, hits: List[dict], max_clusters: int = MAX_CLUSTERS):
        k = max(2, min(max_clusters, len(hits) // CHUNKS_PER_BATCH))
        labels = kmeans([h["embedding"] for h in hits], k)
        groups: dict = {}
        for h, lab in zip(hits, labels):
            groups.setdefault(lab, []).append(h)
        self.n_clusters = len(groups)
        ordered = sorted(groups.values(), key=len, reverse=True)  # start from the largest group
        centroids = [np.mean([h["embedding"] for h in g], axis=0) for g in ordered]
        self._all = [list(ordered[i]) for i in farthest_first_order(centroids)]
        self._queues: List[Deque[dict]] = []
        self._cursor = 0
        self._refill()

    def _refill(self) -> None:
        self._queues = []
        for g in self._all:
            members = list(g)
            random.shuffle(members)
            self._queues.append(deque(members))

    def next_batch(self) -> List[dict]:
        total = sum(len(g) for g in self._all)
        size = min(CHUNKS_PER_BATCH, total)
        batch: List[dict] = []
        while len(batch) < size:
            if not any(self._queues):
                self._refill()
            q = self._queues[self._cursor % len(self._queues)]
            self._cursor += 1
            if q:
                chunk = q.popleft()
                if not any(chunk is b for b in batch):  # a mid-batch refill must not repeat a chunk
                    batch.append(chunk)
        return batch


def make_cycler(hits: List[dict], strategy: Optional[str] = None):
    """Pick the chunk-selection strategy. Falls back to random when coverage can't help:
    the document has no more chunks than one batch, or embeddings are missing."""
    strategy = (strategy or config.SELECTION_STRATEGY).lower()
    if strategy not in STRATEGIES:
        raise ValueError(f"selection strategy must be one of {STRATEGIES}")
    if (strategy == "coverage" and len(hits) > CHUNKS_PER_BATCH
            and all(h.get("embedding") is not None for h in hits)):
        return CoverageCycler(hits)
    return BatchCycler(hits)
