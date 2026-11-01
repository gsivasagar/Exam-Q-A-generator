"""Retrieval benchmark: recall@k and MRR for chunking x retrieval-mode x embedder combinations.

Needs no LLM, only the embedding model (default: built-in MiniLM, runs offline).
A retrieved chunk counts as relevant if it contains the query's gold snippet (eval/retrieval_queries.json).

Usage:
  python scripts/eval_retrieval.py                       # default matrix, writes eval/results/<date>-retrieval.json
  python scripts/eval_retrieval.py --embedder ollama:bge-m3 --embedder default
  python scripts/eval_retrieval.py --reranker fastembed:BAAI/bge-reranker-base      (needs `pip install fastembed`)
  python scripts/eval_retrieval.py --scale real          # real chunk sizes (use with long documents)

By default chunk sizes are scaled down ("small") so these short notes behave like long documents."""
import argparse
import json
import random
import re
import shutil
import statistics
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config, vector_store  # noqa: E402
from app.qa_generator import ingest_document  # noqa: E402
from app.retrieval import retrieve  # noqa: E402

SCALES = {  # words for the legacy chunker; tokens for the structured one
    "small": dict(CHUNK_WORDS=60, CHUNK_OVERLAP_WORDS=10, CHUNK_TARGET_TOKENS=90, CHUNK_OVERLAP_TOKENS=15,
                  CHUNK_MIN_TOKENS=20),
    "real": dict(CHUNK_WORDS=300, CHUNK_OVERLAP_WORDS=50, CHUNK_TARGET_TOKENS=400, CHUNK_OVERLAP_TOKENS=60,
                 CHUNK_MIN_TOKENS=80),
}
MATRIX = [  # name, chunking, retrieval mode, parent expansion
    ("legacy + dense (current default)", "legacy", "dense", False),
    ("legacy + hybrid", "legacy", "hybrid", False),
    ("structured + dense", "structured", "dense", False),
    ("structured + hybrid", "structured", "hybrid", False),
    ("structured + dense + parent", "structured", "dense", True),
    ("structured + hybrid + parent", "structured", "hybrid", True),
]
KS = (1, 3, 5, 10)


def norm(s):
    return re.sub(r"\s+", " ", s.lower())


def run_config(name, chunking, mode, parent, queries, docs, scale, embedder, reranker):
    tmp = tempfile.mkdtemp(prefix="eval_ret_")
    saved = {k: getattr(config, k) for k in ("CHROMA_DIR", "CHUNKING", "PARENT_EXPANSION", "RETRIEVAL_MODE",
                                              "EMBEDDING_MODEL", "RERANKER", *SCALES[scale])}
    try:
        config.CHROMA_DIR = tmp
        config.CHUNKING, config.PARENT_EXPANSION, config.RETRIEVAL_MODE = chunking, parent, mode
        config.EMBEDDING_MODEL, config.RERANKER = embedder, reranker
        for k, v in SCALES[scale].items():
            setattr(config, k, v)
        vector_store._collection.cache_clear()
        n_chunks = sum(ingest_document(None, "eval", filename=p.name, data=p.read_bytes()) for p in docs)
        ranks, lat = [], []
        for q in queries:
            t0 = time.perf_counter()
            hits = retrieve(q["query"], "eval", k=max(KS))
            lat.append((time.perf_counter() - t0) * 1000)
            gold = norm(q["snippet"])
            ranks.append(next((i + 1 for i, h in enumerate(hits) if gold in norm(h["text"])), None))
        return {"name": name, "chunking": chunking, "mode": mode, "parent": parent, "embedder": embedder,
                "reranker": reranker, "scale": scale, "chunks": n_chunks, "ranks": ranks,
                "latency_ms": statistics.mean(lat)}
    finally:
        for k, v in saved.items():
            setattr(config, k, v)
        vector_store._collection.cache_clear()
        shutil.rmtree(tmp, ignore_errors=True)


def metrics(ranks):
    n = len(ranks)
    out = {f"recall@{k}": sum(1 for r in ranks if r and r <= k) / n for k in KS}
    out["mrr"] = sum(1 / r for r in ranks if r) / n
    return out


def paired_delta(base, other, iters=2000, seed=1):
    """Mean change in reciprocal rank (other - base) with a paired-bootstrap 95% CI, plus wins/losses."""
    rr = lambda r: 1 / r if r else 0.0  # noqa: E731
    d = [rr(o) - rr(b) for b, o in zip(base, other)]
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choice(d) for _ in d) for _ in d and range(iters))
    wins, losses = sum(x > 0 for x in d), sum(x < 0 for x in d)
    return statistics.mean(d), means[int(0.025 * iters)], means[int(0.975 * iters)], wins, losses


def bootstrap_ci(ranks, iters=1000, seed=0):
    rng = random.Random(seed)
    vals = sorted(metrics([rng.choice(ranks) for _ in ranks])["mrr"] for _ in range(iters))
    return vals[int(0.025 * iters)], vals[int(0.975 * iters)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", choices=list(SCALES), default="small")
    ap.add_argument("--embedder", action="append", help="repeatable; default: the built-in MiniLM")
    ap.add_argument("--reranker", default="none")
    ap.add_argument("--out", default=None, help="results file (default eval/results/<date>-retrieval.json)")
    args = ap.parse_args()

    spec = json.loads((ROOT / "eval" / "retrieval_queries.json").read_text())
    queries = spec["queries"]
    docs = sorted((ROOT / "eval" / "docs").glob("*.md"))
    results = []
    for emb in args.embedder or ["default"]:
        for name, chunking, mode, parent in MATRIX:
            results.append(run_config(name, chunking, mode, parent, queries, docs, args.scale, emb, args.reranker))

    print(f"{len(queries)} queries, {len(docs)} documents, scale={args.scale}, reranker={args.reranker}")
    kinds = sorted({q["kind"] for q in queries})
    for emb in args.embedder or ["default"]:
        print(f"\nembedder: {emb}")
        print(f"{'configuration':<34}{'chunks':>7}" + "".join(f"{'R@' + str(k):>7}" for k in KS)
              + f"{'MRR':>7}{'95% CI':>14}" + "".join(f"{'MRR-' + k[:4]:>10}" for k in kinds) + f"{'ms/q':>7}")
        for r in (x for x in results if x["embedder"] == emb):
            m = metrics(r["ranks"])
            lo, hi = bootstrap_ci(r["ranks"])
            per_kind = [metrics([rk for rk, q in zip(r["ranks"], queries) if q["kind"] == k])["mrr"] for k in kinds]
            print(f"{r['name']:<34}{r['chunks']:>7}" + "".join(f"{m[f'recall@{k}']:>7.2f}" for k in KS)
                  + f"{m['mrr']:>7.3f}{f'[{lo:.2f},{hi:.2f}]':>14}" + "".join(f"{v:>10.3f}" for v in per_kind)
                  + f"{r['latency_ms']:>7.0f}")
            r["metrics"] = m
            r["mrr_ci95"] = [lo, hi]
            r["mrr_by_kind"] = dict(zip(kinds, per_kind))
        rows = [x for x in results if x["embedder"] == emb]
        base = rows[0]
        print("\n  paired change in MRR vs the first row (95% CI; wins/losses over queries):")
        for r in rows[1:]:
            d, lo, hi, w, l_ = paired_delta(base["ranks"], r["ranks"])
            verdict = "better" if lo > 0 else "worse" if hi < 0 else "no clear difference"
            r["delta_mrr_vs_baseline"] = {"mean": d, "ci95": [lo, hi], "wins": w, "losses": l_, "verdict": verdict}
            print(f"  {r['name']:<34}{d:+.3f}  [{lo:+.3f},{hi:+.3f}]  {w} better / {l_} worse   -> {verdict}")
        print("  NOTE: '+ parent' rows return ~3 chunks of text per hit, so containing the gold snippet is easier by "
              "construction; their gain is not a like-for-like retrieval improvement.")
    out = Path(args.out).resolve() if args.out else ROOT / "eval" / "results" / f"{date.today()}-retrieval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"date": str(date.today()), "scale": args.scale, "n_queries": len(queries),
                               "results": results}, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
