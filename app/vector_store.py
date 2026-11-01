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


def _hits(docs, metas, embeddings=None, expand: bool = False) -> List[Hit]:
    """Turn Chroma results into hits. With `expand`, a chunk is replaced by its stored parent section
    (de-duplicated), the "small-to-big" step: search on small chunks, show the model the larger context."""
    hits, seen_parents = [], set()
    for i, (d, m) in enumerate(zip(docs, metas)):
        m = m or {}
        text = d
        if expand and m.get("parent"):
            if m["parent"] in seen_parents:
                continue
            seen_parents.add(m["parent"])
            text = m["parent"]
        h = {"text": text, "source": m.get("source", ""), "page": m.get("page")}
        if m.get("page_end"):
            h["page_end"] = m["page_end"]
        if m.get("heading"):
            h["heading"] = m["heading"]
        if embeddings is not None:
            h["embedding"] = embeddings[i]
        hits.append(h)
    return hits


def similarity_search(query: str, k: int = 5, doc_id: Optional[str] = None) -> List[Hit]:
    kwargs = {"where": {"doc_id": doc_id}} if doc_id else {}
    res = _collection().query(query_texts=[query], n_results=k,
                              include=["documents", "metadatas", "embeddings"], **kwargs)
    if not res["documents"]:
        return []
    embeddings = res["embeddings"][0] if res.get("embeddings") is not None else None
    return _hits(res["documents"][0], res["metadatas"][0], embeddings, expand=config.PARENT_EXPANSION)


def get_all_chunks(doc_id: str) -> List[Hit]:
    res = _collection().get(where={"doc_id": doc_id}, include=["documents", "metadatas", "embeddings"])
    return _hits(res["documents"], res["metadatas"], res.get("embeddings"))


def delete_doc(doc_id: str) -> None:
    _collection().delete(where={"doc_id": doc_id})
