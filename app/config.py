"""Central configuration, read once from environment variables."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:latest")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", str(DATA_DIR / "chroma"))
DB_PATH = Path(os.getenv("QUIZ_DB_PATH", str(DATA_DIR / "quiz_results.db")))

# LLM call behaviour
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "300"))            # seconds per request
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))        # retries on transient errors
LLM_RETRY_BACKOFF = float(os.getenv("LLM_RETRY_BACKOFF", "1"))  # seconds; doubles each retry
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "0")) or None  # unset = Ollama's default window
# Optional Gemini thinking-token cap (0 disables thinking on 2.5 Flash); unset = model default.
GEMINI_THINKING_BUDGET = (int(os.environ["GEMINI_THINKING_BUDGET"])
                          if os.getenv("GEMINI_THINKING_BUDGET", "").strip() else None)

# Embedding model: "default" (Chroma MiniLM) or "ollama:<model>" e.g. ollama:bge-m3
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "default").strip()

# How chunks are chosen for each LLM call: "coverage" (cluster, then round-robin) or "random".
SELECTION_STRATEGY = os.getenv("SELECTION_STRATEGY", "coverage").strip().lower()

# Chunking: "legacy" (300-word sentence chunks, one page each) or "structured" (heading-aware,
# token-sized, may span pages). PARENT_EXPANSION sends the surrounding section for topic searches.
CHUNKING = os.getenv("CHUNKING", "legacy").strip().lower()
CHUNK_TARGET_TOKENS = int(os.getenv("CHUNK_TARGET_TOKENS", "400"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "60"))
PARENT_EXPANSION = os.getenv("PARENT_EXPANSION", "false").strip().lower() in {"1", "true", "yes", "on"}

# Topic retrieval: "dense" (default) or "hybrid" (BM25 + dense, fused). RERANKER: "none" or
# "fastembed:<model>" (optional dependency). RETRIEVAL_MAX_DISTANCE drops dense hits farther than this
# (unset = keep all); with hybrid it also lets a topic that is absent from the notes return nothing.
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "dense").strip().lower()
RERANKER = os.getenv("RERANKER", "none").strip()
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "20"))
RETRIEVAL_MAX_DISTANCE = (float(os.environ["RETRIEVAL_MAX_DISTANCE"])
                          if os.getenv("RETRIEVAL_MAX_DISTANCE", "").strip() else None)

PASS_THRESHOLD = 0.6
