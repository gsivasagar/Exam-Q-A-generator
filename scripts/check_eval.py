"""Fail (exit 1) if evaluation results fall below eval/thresholds.json. Used by the scheduled workflow.

Usage: python scripts/check_eval.py --retrieval R.json --grading G.json --selection S.json   (any subset)"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(p):
    return json.loads(Path(p).read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval")
    ap.add_argument("--grading")
    ap.add_argument("--selection")
    ap.add_argument("--thresholds", default=str(ROOT / "eval" / "thresholds.json"))
    args = ap.parse_args()
    t = load(args.thresholds)
    failures, checks = [], 0

    def check(label, value, floor):
        nonlocal checks
        checks += 1
        status = "ok  " if value >= floor else "FAIL"
        print(f"{status} {label}: {value:.3f} (floor {floor})")
        if value < floor:
            failures.append(label)

    if args.retrieval:
        want = t["retrieval"]
        row = next((r for r in load(args.retrieval)["results"] if r["name"] == want["config"]), None)
        if row is None:
            failures.append(f"retrieval config '{want['config']}' missing")
        else:
            check("retrieval MRR", row["metrics"]["mrr"], want["min_mrr"])
            check("retrieval recall@5", row["metrics"]["recall@5"], want["min_recall@5"])
    if args.grading:
        m = load(args.grading)["metrics"]
        check("stub grading agreement", m["agreement"], t["grading_stub"]["min_agreement"])
        check("pre-checks agreeing with labels", m["precheck_agree"], t["grading_stub"]["min_precheck_agree"])
    if args.selection:
        check("selection document coverage", load(args.selection)["metrics"]["context_doc_coverage"],
              t["selection_stub"]["min_context_doc_coverage"])
    if not checks and not failures:
        sys.exit("nothing to check: pass --retrieval, --grading and/or --selection")
    if failures:
        sys.exit(f"{len(failures)} evaluation check(s) failed: {', '.join(failures)}")
    print("all evaluation checks passed")


if __name__ == "__main__":
    main()
