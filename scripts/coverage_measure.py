"""Compare random vs coverage chunk selection with a stub LLM (no model needed).

Usage (isolated store):  CHROMA_PERSIST_DIR=/tmp/c DATA_DIR=/tmp/c python scripts/coverage_measure.py [pdf_dir]
Treats every PDF in pdf_dir (default ./tmp) as one large document."""
import glob
import json
import os
import random
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "tmp"
sys.path.insert(0, str(ROOT))
from app import qa_generator as q, vector_store as vs
from app.clustering import mean_pairwise_similarity
for p in sorted(glob.glob(str(PDF_DIR / "*.pdf"))):
    q.ingest_pdf(p, "big", filename=os.path.basename(p))
allh = vs.get_all_chunks("big")
emb = {h["text"]: h["embedding"] for h in allh}; src = {h["text"]: h["source"] for h in allh}
print("chunks", len(allh), "sources", len(set(src.values())))
class Rec:
    def __init__(s): s.texts = []; s.k = 0
    def generate(s, prompt, *, schema=None, **_):
        s.k += 1
        ctx = prompt.split("<context>")[-1]
        parts = re.split(r"\n\n(?=\[\d+\] )", ctx.strip().removeprefix("\n"))
        s.texts += [re.sub(r"^\[\d+\] ", "", x.replace("</context>", "").strip()) for x in parts]
        return json.dumps({"questions": [{"question": f"Q{s.k}-{i}?", "answer": f"a{i}", "topic": "T", "chunk": 1 + i % 4}
                                          for i in range(5)]})
def run(strategy, n, seed):
    random.seed(seed); r = Rec()
    q.generate_qa_pairs("big", r, n=n, question_type="short", strategy=strategy)
    texts = [t for t in dict.fromkeys(r.texts) if t in emb]   # unique chunks sent to the model
    return len(texts), len({src[t] for t in texts}), mean_pairwise_similarity([emb[t] for t in texts])
for n in (10, 25):
    for strat in ("random", "coverage"):
        res = [run(strat, n, s) for s in range(30)]
        print(f"n={n:<3} {strat:<9} unique chunks used {st.mean(r[0] for r in res):5.1f} | distinct PDFs {st.mean(r[1] for r in res):5.1f} | mean pairwise sim {st.mean(r[2] for r in res):.3f}")
