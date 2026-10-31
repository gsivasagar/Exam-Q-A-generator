"""ChromaDB-backed chunk store. One collection, filtered by doc_id (= quiz/session)."""
from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Dict, List, Optional

from . import config
from .pdf_loader import Chunk

Hit = Dict  # {"text", "source", "page"}


@lru_cache(maxsize=1)
def _collection():
    import chromadb
    # Chroma's built-in ONNX build of all-MiniLM-L6-v2: same model as before,
    # but no torch / sentence-transformers dependency.
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    ef = DefaultEmbeddingFunction()
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    return client.get_or_create_collection(name="exam_chunks", embedding_function=ef)


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


def _hits(docs, metas) -> List[Hit]:
    return [{"text": d, "source": (m or {}).get("source", ""), "page": (m or {}).get("page")}
            for d, m in zip(docs, metas)]


def similarity_search(query: str, k: int = 5, doc_id: Optional[str] = None) -> List[Hit]:
    kwargs = {"where": {"doc_id": doc_id}} if doc_id else {}
    res = _collection().query(query_texts=[query], n_results=k, **kwargs)
    if not res["documents"]:
        return []
    return _hits(res["documents"][0], res["metadatas"][0])


def get_all_chunks(doc_id: str) -> List[Hit]:
    res = _collection().get(where={"doc_id": doc_id})
    return _hits(res["documents"], res["metadatas"])


def delete_doc(doc_id: str) -> None:
    _collection().delete(where={"doc_id": doc_id})
