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
    _collection().upsert(
        documents=[c.text for c in chunks],
        ids=[f"{doc_id}-{file_id}-{i}" for i in range(len(chunks))],
        metadatas=[{"doc_id": doc_id, "source": source, "file_id": file_id,
                    "chunk": i, "page": c.page} for i, c in enumerate(chunks)],
    )
    return len(chunks)


def _hits(docs, metas, embeddings=None) -> List[Hit]:
    hits = [{"text": d, "source": (m or {}).get("source", ""), "page": (m or {}).get("page")}
            for d, m in zip(docs, metas)]
    if embeddings is not None:
        for h, e in zip(hits, embeddings):
            h["embedding"] = e
    return hits


def similarity_search(query: str, k: int = 5, doc_id: Optional[str] = None) -> List[Hit]:
    kwargs = {"where": {"doc_id": doc_id}} if doc_id else {}
    res = _collection().query(query_texts=[query], n_results=k,
                              include=["documents", "metadatas", "embeddings"], **kwargs)
    if not res["documents"]:
        return []
    embeddings = res["embeddings"][0] if res.get("embeddings") is not None else None
    return _hits(res["documents"][0], res["metadatas"][0], embeddings)


def get_all_chunks(doc_id: str) -> List[Hit]:
    res = _collection().get(where={"doc_id": doc_id}, include=["documents", "metadatas", "embeddings"])
    return _hits(res["documents"], res["metadatas"], res.get("embeddings"))


def delete_doc(doc_id: str) -> None:
    _collection().delete(where={"doc_id": doc_id})
