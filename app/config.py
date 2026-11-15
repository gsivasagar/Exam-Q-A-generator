"""Central configuration, read once from environment variables."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:latest")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# Characters of notes we will put in one Gemini call (its window is far larger; this bounds cost/latency).
GEMINI_CONTEXT_CHARS = int(os.getenv("GEMINI_CONTEXT_CHARS", "120000"))

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", str(DATA_DIR / "chroma"))
DB_PATH = Path(os.getenv("QUIZ_DB_PATH", str(DATA_DIR / "quiz_results.db")))

# Parallel LLM calls. Providers declare their own limit (Ollama OLLAMA_CONCURRENCY, default 1: one local
# model rarely speeds up; set it equal to OLLAMA_NUM_PARALLEL to benefit). LLM_CACHE_DIR enables the
# on-disk response cache (development/evaluation).
LLM_CONCURRENCY = int(os.getenv("LLM_CONCURRENCY", "1"))
OLLAMA_CONCURRENCY = int(os.getenv("OLLAMA_CONCURRENCY", "1"))
GEMINI_CONCURRENCY = int(os.getenv("GEMINI_CONCURRENCY", "4"))
LLM_CACHE_DIR = os.getenv("LLM_CACHE_DIR", "").strip() or None

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
CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", "300"))              # legacy chunker: words per chunk
CHUNK_OVERLAP_WORDS = int(os.getenv("CHUNK_OVERLAP_WORDS", "50"))
CHUNK_MIN_TOKENS = int(os.getenv("CHUNK_MIN_TOKENS", "80"))      # structured chunker: smallest standalone chunk
CHUNK_TARGET_TOKENS = int(os.getenv("CHUNK_TARGET_TOKENS", "400"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "60"))
PARENT_EXPANSION = os.getenv("PARENT_EXPANSION", "false").strip().lower() in {"1", "true", "yes", "on"}

# Topic retrieval: "auto" (default: hybrid for documents of at least HYBRID_MIN_CHUNKS chunks, dense below that),
# "dense" or "hybrid" (BM25 + dense, fused). RERANKER: "none" or "fastembed:<model>" (optional dependency).
# RETRIEVAL_MAX_DISTANCE drops dense hits farther than this (unset = keep all); with hybrid it also lets a
# topic that is absent from the notes return nothing.
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "auto").strip().lower()
HYBRID_MIN_CHUNKS = int(os.getenv("HYBRID_MIN_CHUNKS", "20"))
RERANKER = os.getenv("RERANKER", "none").strip()
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "20"))
RETRIEVAL_MAX_DISTANCE = (float(os.environ["RETRIEVAL_MAX_DISTANCE"])
                          if os.getenv("RETRIEVAL_MAX_DISTANCE", "").strip() else None)

# Adaptive routing: when the whole document fits the context budget, send all of it instead of
# selecting chunks. FULL_CONTEXT_BATCH is how many questions to ask per call in that mode.
ADAPTIVE_ROUTING = os.getenv("ADAPTIVE_ROUTING", "on").strip().lower() in {"1", "true", "yes", "on"}
FULL_CONTEXT_MAX_CHARS = int(os.getenv("FULL_CONTEXT_MAX_CHARS", "6000"))
FULL_CONTEXT_BATCH = int(os.getenv("FULL_CONTEXT_BATCH", "5"))

# Question quality. VERIFY_QUESTIONS adds a second LLM pass that drops questions the cited passage does
# not support (extra calls). SEMANTIC_DEDUPE drops paraphrases using embedding similarity.
VERIFY_QUESTIONS = os.getenv("VERIFY_QUESTIONS", "off").strip().lower() in {"1", "true", "yes", "on"}
SEMANTIC_DEDUPE = os.getenv("SEMANTIC_DEDUPE", "off").strip().lower() in {"1", "true", "yes", "on"}
SEMANTIC_DEDUPE_THRESHOLD = float(os.getenv("SEMANTIC_DEDUPE_THRESHOLD", "0.85"))

# Grading: "rubric" scores short answers by key points covered (when the question has them; computed in
# code), "holistic" asks the model for a 0-1 score directly.
GRADING_MODE = os.getenv("GRADING_MODE", "rubric").strip().lower()

# Uploads and ingestion. OCR_MODE "auto" (default) runs OCR only for PDFs with no text layer, when the
# optional OCR packages and Tesseract are installed; "off" never does.
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "25"))
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "300"))
URL_TIMEOUT = float(os.getenv("URL_TIMEOUT", "15"))
OCR_MODE = os.getenv("OCR_MODE", "auto").strip().lower()
# Reading scanned PDFs and images with Gemini costs one model call per page, so a scan longer than this is
# refused with a clear message instead of silently truncated or quietly using up the quota.
OCR_MAX_PAGES = int(os.getenv("OCR_MAX_PAGES", "30"))

# Uploaded documents are forgotten from the vector store after this many days (0 = never).
DOC_TTL_DAYS = float(os.getenv("DOC_TTL_DAYS", "7"))

PASS_THRESHOLD = 0.6


# Accounts and the HTTP API. AUTH_SECRET signs access tokens: set a long random value in production (the API
# refuses to start without one when APP_ENV=production). Leaving it unset in development uses a throwaway
# secret, so tokens stop working whenever the server restarts.
APP_ENV = os.getenv("APP_ENV", "development").strip().lower()
AUTH_SECRET = os.getenv("AUTH_SECRET", "")
ACCESS_TTL_MIN = int(os.getenv("ACCESS_TTL_MIN", "15"))
REFRESH_TTL_DAYS = int(os.getenv("REFRESH_TTL_DAYS", "30"))
# A session cookie (no "keep me signed in") is also refused after this many hours on the server side.
SESSION_TTL_HOURS = int(os.getenv("SESSION_TTL_HOURS", "12"))
GUEST_TTL_DAYS = int(os.getenv("GUEST_TTL_DAYS", "7"))
GUEST_MAX_FILES = int(os.getenv("GUEST_MAX_FILES", "3"))
GUEST_MAX_UPLOAD_MB = float(os.getenv("GUEST_MAX_UPLOAD_MB", "10"))
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "0" if APP_ENV != "production" else "1") == "1"
APP_ORIGIN = os.getenv("APP_ORIGIN", "http://localhost:5173").rstrip("/")
# How many reverse proxies sit in front of the API (0 = none). The client address is read from X-Forwarded-For
# only that many hops from the right, so a visitor cannot make up their own address.
TRUSTED_PROXY_HOPS = int(os.getenv("TRUSTED_PROXY_HOPS", "0"))
