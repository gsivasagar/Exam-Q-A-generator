"""Simulate the effect of parallel LLM calls with a fake model that sleeps per call (no model needed).

Usage: CHROMA_PERSIST_DIR=/tmp/c DATA_DIR=/tmp/c python scripts/latency_sim.py [pdf_dir]"""
import glob
import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "tmp"
sys.path.insert(0, str(ROOT))
from app import qa_generator as q
from app.grader import grade_all
q.ingest_pdf(sorted(glob.glob(str(PDF_DIR / "*.pdf")))[0], "d", filename="a.pdf")
DELAY = 0.5   # pretend each LLM call takes 0.5 s
class Slow:
    def __init__(s, workers): s.max_concurrency = workers; s.k = 0; s.lock = threading.Lock()
    def generate(s, prompt, *, schema=None, **_):
        with s.lock: s.k += 1; k = s.k
        time.sleep(DELAY)
        if "questions" in schema.get("properties", {}):
            return json.dumps({"questions": [{"question": f"Q{k}-{i} unique?", "answer": "a", "topic": "T", "chunk": 1} for i in range(5)]})
        return '{"score": 0.7, "feedback": "ok"}'
print(f"simulated call latency {DELAY}s; 10-question short-answer quiz = generate + grade 10 answers")
base = None
for w in (1, 2, 4, 8):
    llm = Slow(w); t0 = time.perf_counter()
    qs = q.generate_qa_pairs("d", llm, n=10)
    t1 = time.perf_counter()
    grade_all(qs, ["my answer"] * len(qs), llm, workers=w)
    t2 = time.perf_counter()
    total = t2 - t0; base = base or total
    print(f"workers={w}: generate {t1-t0:4.1f}s  grade {t2-t1:4.1f}s  total {total:4.1f}s  speedup x{base/total:.1f}  calls={llm.k}")
