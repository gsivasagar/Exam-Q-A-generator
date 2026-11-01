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

    def take(self) -> Optional[dict]:
        """The next unused chunk in rotation order, or None once every chunk has been handed out this round."""
        for _ in range(len(self._queues)):
            q = self._queues[self._cursor % len(self._queues)]
            self._cursor += 1
            if q:
                return q.popleft()
        return None

    def reset(self) -> None:
        """Start a new round: every chunk becomes available again."""
        self._refill()

    def next_batch(self) -> List[dict]:
        total = sum(len(g) for g in self._all)
        size = min(CHUNKS_PER_BATCH, total)
        batch: List[dict] = []
        while len(batch) < size:
            chunk = self.take()
            if chunk is None:
                self._refill()
            elif not any(chunk is b for b in batch):  # a mid-batch refill must not repeat a chunk
                batch.append(chunk)
        return batch


class _ShuffledSource:
    """A small source (too few chunks to cluster): its chunks in shuffled order, one round at a time."""

    def __init__(self, hits: List[dict]):
        self._hits = list(hits)
        self.reset()

    def reset(self) -> None:
        items = list(self._hits)
        random.shuffle(items)
        self._queue: Deque[dict] = deque(items)

    def take(self) -> Optional[dict]:
        return self._queue.popleft() if self._queue else None


class MultiSourceCycler:
    """Coverage selection across several uploaded files: each batch takes chunks from different FILES first
    (round-robin), and within a file uses meaning-based clusters. So a quiz built from five files draws on
    all of them, and one huge file cannot crowd out small ones. A file that runs out of unused chunks is
    skipped until every file is used up; then a new round begins."""

    def __init__(self, hits: List[dict], max_clusters: int = MAX_CLUSTERS):
        by_source: dict = {}
        for h in hits:
            by_source.setdefault(h.get("source", ""), []).append(h)
        sources = list(by_source.values())
        random.shuffle(sources)  # which file leads is random, so quizzes vary
        self.n_sources = len(sources)
        self._total = len(hits)
        self._sources = [CoverageCycler(g, max_clusters) if len(g) > CHUNKS_PER_BATCH else _ShuffledSource(g)
                         for g in sources]
        self._cursor = 0

    def next_batch(self) -> List[dict]:
        size = min(CHUNKS_PER_BATCH, self._total)
        batch: List[dict] = []
        while len(batch) < size:
            progressed = False
            for _ in range(len(self._sources)):
                src = self._sources[self._cursor % len(self._sources)]
                self._cursor += 1
                chunk = src.take()
                if chunk is not None and not any(chunk is b for b in batch):
                    batch.append(chunk)
                    progressed = True
                    if len(batch) == size:
                        break
            if not progressed:  # every file is used up: begin a new round
                for src in self._sources:
                    src.reset()
        return batch


def make_cycler(hits: List[dict], strategy: Optional[str] = None):
    """Pick the chunk-selection strategy. Falls back to random when coverage can't help:
    the document has no more chunks than one batch, or embeddings are missing."""
    strategy = (strategy or config.SELECTION_STRATEGY).lower()
    if strategy not in STRATEGIES:
        raise ValueError(f"selection strategy must be one of {STRATEGIES}")
    if (strategy == "coverage" and len(hits) > CHUNKS_PER_BATCH
            and all(h.get("embedding") is not None for h in hits)):
        if len({h.get("source", "") for h in hits}) > 1:
            return MultiSourceCycler(hits)
        return CoverageCycler(hits)
    return BatchCycler(hits)


class FullContextCycler:
    """Router path "full": every call sees the whole document, in reading order."""

    def __init__(self, hits: List[dict]):
        self.hits = hits

    def next_batch(self) -> List[dict]:
        return list(self.hits)
