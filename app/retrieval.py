"""Topic retrieval: dense search, or hybrid (BM25 + dense fused with RRF), with optional reranking.

RETRIEVAL_MODE=dense  (default) the original behaviour: top-k nearest chunks by embedding.
RETRIEVAL_MODE=hybrid           BM25 keyword ranking + dense ranking, fused with Reciprocal Rank Fusion.
                                Catches exact terms (formulas, acronyms, names) that embeddings miss and
                                returns nothing for a topic that does not appear in the document.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from . import config
from .bm25 import BM25
from .rerank import Reranker, get_reranker
from .vector_store import get_all_chunks, similarity_search

RRF_K = 60
MODES = ("dense", "hybrid")


def rrf_fuse(rankings: Sequence[Sequence[int]], k: int = RRF_K) -> List[int]:
    """Reciprocal Rank Fusion: merge several best-first rankings of item ids into one."""
    score: Dict[int, float] = {}
    for ranking in rankings:
        for pos, item in enumerate(ranking):
            score[item] = score.get(item, 0.0) + 1.0 / (k + pos + 1)
    return sorted(score, key=lambda i: (-score[i], i))


def _dense(query: str, doc_id: str, k: int) -> List[dict]:
    hits = similarity_search(query, k=k, doc_id=doc_id)
    limit = config.RETRIEVAL_MAX_DISTANCE
    if limit is not None:
        hits = [h for h in hits if h.get("distance") is None or h["distance"] <= limit]
    return hits


def retrieve(query: str, doc_id: str, *, k: int = 30, mode: Optional[str] = None,
             reranker: Optional[Reranker] = None, candidates: Optional[int] = None) -> List[dict]:
    """Chunks relevant to `query` in one document set, best first (at most k)."""
    mode = (mode or config.RETRIEVAL_MODE).lower()
    if mode not in MODES:
        raise ValueError(f"retrieval mode must be one of {MODES}")
    reranker = reranker if reranker is not None else get_reranker()

    if mode == "dense":
        hits = _dense(query, doc_id, k if reranker is None else max(k, candidates or config.RERANK_CANDIDATES))
    else:
        pool = get_all_chunks(doc_id)
        if not pool:
            return []
        by_text = {h["text"]: h for h in pool}
        texts = list(by_text)
        keyword = BM25(texts).rank(query, k=len(texts))
        dense_hits = _dense(query, doc_id, min(len(texts), max(k, 10)))
        index = {t: i for i, t in enumerate(texts)}
        dense_rank = [index[h["text"]] for h in dense_hits if h["text"] in index]
        fused = rrf_fuse([keyword, dense_rank])
        # With a distance cutoff set, a topic with no keyword match and no close embedding is "not present".
        if not keyword and config.RETRIEVAL_MAX_DISTANCE is not None and not dense_rank:
            return []
        hits = [by_text[texts[i]] for i in fused]

    if reranker is not None and len(hits) > 1:
        hits = reranker.rerank(query, hits[: max(k, candidates or config.RERANK_CANDIDATES)])
    return hits[:k]
