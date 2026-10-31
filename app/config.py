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

PASS_THRESHOLD = 0.6
