"""Compare two evaluation result files (from eval_retrieval.py or eval_generation.py).

Usage: python scripts/compare_eval.py before.json after.json [--config "legacy + dense (current default)"]

Retrieval files: paired bootstrap of the change in MRR per configuration (same queries in the same order),
with the change in recall@k and latency. Generation files: side-by-side metric table with the change."""
import argparse
import json
import random
import statistics
import sys
from pathlib import Path


def rr(r):
    return 1 / r if r else 0.0


def paired(base, other, iters=2000, seed=1):
    d = [rr(o) - rr(b) for b, o in zip(base, other)]
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choice(d) for _ in d) for _ in range(iters))
    return statistics.mean(d), means[int(0.025 * iters)], means[int(0.975 * iters)], sum(x > 0 for x in d), sum(x < 0 for x in d)


def compare_retrieval(a, b, only):
    ra = {r["name"]: r for r in a["results"] if r["embedder"] == a["results"][0]["embedder"]}
    rb = {r["name"]: r for r in b["results"] if r["embedder"] == b["results"][0]["embedder"]}
    print(f"before: reranker={a['results'][0]['reranker']} embedder={a['results'][0]['embedder']}")
    print(f"after:  reranker={b['results'][0]['reranker']} embedder={b['results'][0]['embedder']}\n")
    print(f"{'configuration':<34}{'MRR before':>11}{'after':>8}{'change':>9}{'95% CI':>18}{'better/worse':>14}{'ms/q':>14}")
    for name in ra:
        if name not in rb or (only and name != only):
            continue
        if len(ra[name]["ranks"]) != len(rb[name]["ranks"]):
            print(f"{name}: different query sets, skipped")
            continue
        d, lo, hi, w, l_ = paired(ra[name]["ranks"], rb[name]["ranks"])
        verdict = "BETTER" if lo > 0 else "WORSE" if hi < 0 else "~same"
        print(f"{name:<34}{ra[name]['metrics']['mrr']:>11.3f}{rb[name]['metrics']['mrr']:>8.3f}{d:>+9.3f}"
              f"{f'[{lo:+.3f},{hi:+.3f}]':>18}{f'{w}/{l_}':>14}{ra[name]['latency_ms']:>7.0f}->{rb[name]['latency_ms']:<5.0f} {verdict}")


def flatten(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from flatten(v, f"{prefix}{k}.")
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            yield f"{prefix}{k}", v


def compare_generation(a, b):
    fa, fb = dict(flatten(a.get("metrics", {}))), dict(flatten(b.get("metrics", {})))
    print(f"before: {a.get('provider')} {a.get('model')} ({a.get('date')})\nafter:  {b.get('provider')} {b.get('model')} ({b.get('date')})\n")
    print(f"{'metric':<44}{'before':>12}{'after':>12}{'change':>12}")
    for k in sorted(set(fa) | set(fb)):
        x, y = fa.get(k), fb.get(k)
        ch = "" if x is None or y is None else f"{y - x:+.3f}"
        print(f"{k:<44}{'' if x is None else format(x, '.3f'):>12}{'' if y is None else format(y, '.3f'):>12}{ch:>12}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    a, b = (json.loads(Path(p).read_text()) for p in (args.before, args.after))
    if "results" in a and "results" in b:
        compare_retrieval(a, b, args.config)
    elif "metrics" in a and "metrics" in b:
        compare_generation(a, b)
    else:
        sys.exit("The two files are not the same kind of evaluation result.")


if __name__ == "__main__":
    main()
