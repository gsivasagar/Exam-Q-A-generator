from typing import Optional, List
from pathlib import Path
from .db import vectordb
from typing import List
import chromadb
import os
from sentence_transformers import SentenceTransformer

EMB_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", ".chromadb")
_client = chromadb.PersistentClient(path=CHROMA_DIR)
_collection = _client.get_or_create_collection("pdf_chunks")
_embedder = SentenceTransformer(EMB_MODEL_NAME)

def add_chunks(chunks: List[str], doc_id: str):
    vectordb.add(
        documents=chunks,
        ids=[f"{doc_id}-{i}" for i in range(len(chunks))],
        metadatas=[{"doc_id": doc_id}] * len(chunks)
    )

def similarity_search(query: str, k: int = 5, doc_id: Optional[str] = None) -> List[str]:
    """
    Perform semantic similarity search using ChromaDB.
    Returns top `k` most relevant document chunks.
    """
    results = vectordb.query(
        query_texts=[query],
        n_results=k,
        where={"doc_id": doc_id} if doc_id else {}
    )
    return results["documents"][0] if results["documents"] else []