"""ChromaDB-backed chunk store. One collection, filtered by doc_id (= quiz/session)."""
from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import List, Optional

from . import config


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


def add_chunks(chunks: List[str], doc_id: str, source: str, file_id: str) -> int:
    """Store chunks. IDs include file_id so multiple PDFs never collide; re-adding
    the same file is idempotent (upsert)."""
    if not chunks:
        return 0
    _collection().upsert(
        documents=chunks,
        ids=[f"{doc_id}-{file_id}-{i}" for i in range(len(chunks))],
        metadatas=[{"doc_id": doc_id, "source": source, "file_id": file_id, "chunk": i}
                   for i in range(len(chunks))],
    )
    return len(chunks)


def similarity_search(query: str, k: int = 5, doc_id: Optional[str] = None) -> List[str]:
    kwargs = {"where": {"doc_id": doc_id}} if doc_id else {}
    col = _collection()
    results = col.query(query_texts=[query], n_results=k, **kwargs)
    return results["documents"][0] if results["documents"] else []


def get_all_chunks(doc_id: str) -> List[str]:
    return _collection().get(where={"doc_id": doc_id})["documents"]


def delete_doc(doc_id: str) -> None:
    _collection().delete(where={"doc_id": doc_id})
