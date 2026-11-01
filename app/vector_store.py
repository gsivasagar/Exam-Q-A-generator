"""ChromaDB-backed chunk store. One collection, filtered by doc_id (= quiz/session)."""
from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Dict, List, Optional

from . import config
from .embeddings import collection_name, get_embedding_function
from .pdf_loader import Chunk

Hit = Dict  # {"text", "source", "page"}


@lru_cache(maxsize=4)
def _open(path: str, spec: str):
    import chromadb

    client = chromadb.PersistentClient(path=path)
    return client.get_or_create_collection(name=collection_name(spec),
                                           embedding_function=get_embedding_function(spec))


def _collection():
    """The collection for the *current* config. Cached per (path, embedder), so changing either
    at runtime opens the right store instead of silently reusing a stale one."""
    return _open(config.CHROMA_DIR, config.EMBEDDING_MODEL)


_collection.cache_clear = _open.cache_clear  # type: ignore[attr-defined]  # tests/fixtures reset the cache


def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def add_chunks(chunks: List[Chunk], doc_id: str, source: str, file_id: str) -> int:
    """Store chunks. IDs include file_id so multiple PDFs never collide; re-adding
    the same file is idempotent (upsert)."""
    if not chunks:
        return 0
    parents = _parent_texts(chunks) if config.PARENT_EXPANSION else None
    metadatas = []
    for i, c in enumerate(chunks):
        meta = {"doc_id": doc_id, "source": source, "file_id": file_id, "chunk": i, "page": c.page}
        if c.heading:
            meta["heading"] = c.heading
        if c.page_end:
            meta["page_end"] = c.page_end
        if parents and parents[i] != c.text:
            meta["parent"] = parents[i]
        metadatas.append(meta)
    _collection().upsert(
        documents=[c.text for c in chunks],
        ids=[f"{doc_id}-{file_id}-{i}" for i in range(len(chunks))],
        metadatas=metadatas,
    )
    return len(chunks)


def _parent_texts(chunks: List[Chunk], max_chars: int = 3600) -> List[str]:
    """For small-to-big retrieval: each chunk plus its neighbours under the same heading (capped)."""
    out = []
    for i, c in enumerate(chunks):
        parts = [c.text]
        for j in (i - 1, i + 1):
            if 0 <= j < len(chunks) and chunks[j].heading == c.heading and chunks[j].page - c.page in (-1, 0, 1):
                parts.insert(0, chunks[j].text) if j < i else parts.append(chunks[j].text)
        text = " ".join(parts)
        out.append(text if len(text) <= max_chars else c.text)
    return out


def expand_parents(hits: List[Hit]) -> List[Hit]:
    """The "small-to-big" step: replace each chunk by its stored parent section (when it has one),
    keeping order and dropping repeats. Hits without a parent are unchanged."""
    out, seen = [], set()
    for h in hits:
        parent = h.get("parent")
        if parent:
            if parent in seen:
                continue
            seen.add(parent)
            h = {**h, "text": parent}
        out.append(h)
    return out


def _hits(docs, metas, embeddings=None) -> List[Hit]:
    hits = []
    for i, (d, m) in enumerate(zip(docs, metas)):
        m = m or {}
        h = {"text": d, "source": m.get("source", ""), "page": m.get("page")}
        if m.get("parent"):
            h["parent"] = m["parent"]
        if m.get("page_end"):
            h["page_end"] = m["page_end"]
        if m.get("heading"):
            h["heading"] = m["heading"]
        if embeddings is not None:
            h["embedding"] = embeddings[i]
        hits.append(h)
    return hits


def similarity_search(query: str, k: int = 5, doc_id: Optional[str] = None,
                      expand: Optional[bool] = None) -> List[Hit]:
    """Nearest chunks to `query`. `expand` (default: PARENT_EXPANSION) swaps chunks for their parent sections."""
    kwargs = {"where": {"doc_id": doc_id}} if doc_id else {}
    res = _collection().query(query_texts=[query], n_results=k,
                              include=["documents", "metadatas", "embeddings", "distances"], **kwargs)
    if not res["documents"]:
        return []
    embeddings = res["embeddings"][0] if res.get("embeddings") is not None else None
    hits = _hits(res["documents"][0], res["metadatas"][0], embeddings)
    for h, d in zip(hits, (res.get("distances") or [[]])[0]):
        h["distance"] = float(d)
    return expand_parents(hits) if (config.PARENT_EXPANSION if expand is None else expand) else hits


def get_all_chunks(doc_id: str) -> List[Hit]:
    res = _collection().get(where={"doc_id": doc_id}, include=["documents", "metadatas", "embeddings"])
    return _hits(res["documents"], res["metadatas"], res.get("embeddings"))


def delete_doc(doc_id: str) -> None:
    _collection().delete(where={"doc_id": doc_id})
