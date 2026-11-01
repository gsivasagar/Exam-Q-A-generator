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

# Uploaded documents are forgotten from the vector store after this many days (0 = never).
DOC_TTL_DAYS = float(os.getenv("DOC_TTL_DAYS", "7"))

PASS_THRESHOLD = 0.6
