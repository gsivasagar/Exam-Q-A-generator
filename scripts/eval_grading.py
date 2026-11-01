"""Grading evaluation on eval/grading_calibration.json (33 hand-labelled answers).

Reports: how many answers the LLM-free pre-checks settle (and whether they are right), agreement with the
labels, mean score per label, error vs target score, repeatability across repeated runs, and ungraded rate.

  python scripts/eval_grading.py --provider stub                 # lexical stand-in, no model
  python scripts/eval_grading.py --provider ollama --model gemma3 --repeats 3
  python scripts/eval_grading.py --provider gemini --mode holistic

--mode rubric (default) uses each item's key points; holistic ignores them."""
import argparse
import json
import statistics
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _evalkit import ROOT, Recorder, make_provider  # noqa: E402

from app import config  # noqa: E402
from app.grader import grade_question, precheck  # noqa: E402


def label_of(score):
    return None if score is None else "correct" if score >= 0.75 else "partial" if score >= 0.25 else "wrong"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", choices=["stub", "ollama", "gemini"], default="stub")
    ap.add_argument("--model", default=None)
    ap.add_argument("--mode", choices=["rubric", "holistic"], default="rubric")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    spec = json.loads((ROOT / "eval" / "grading_calibration.json").read_text())
    items, targets = spec["items"], spec["targets"]
    config.GRADING_MODE = args.mode

    # 1. LLM-free pre-checks
    settled = [(it, precheck(it["reference"], it["student"])) for it in items]
    settled = [(it, r) for it, r in settled if r is not None]
    right = sum(1 for it, r in settled if label_of(r["score"]) == it["label"])
    print(f"pre-checks settle {len(settled)}/{len(items)} answers without a model; {right}/{len(settled)} agree with the labels")

    # 2. grading with the provider
    rec = Recorder(make_provider(args.provider, args.model, task="grading"))
    scores = {it["id"]: [] for it in items}
    for _ in range(args.repeats):
        for it in items:
            qa = {"type": "short", "question": it["question"], "answer": it["reference"],
                  "key_points": it["key_points"]}
            scores[it["id"]].append(grade_question(qa, it["student"], rec)["score"])
    firsts = {k: v[0] for k, v in scores.items()}
    graded = {k: v for k, v in firsts.items() if v is not None}
    agree = sum(1 for it in items if label_of(firsts[it["id"]]) == it["label"])
    per_label = {lab: statistics.mean(graded[it["id"]] for it in items if it["label"] == lab and it["id"] in graded)
                 for lab in targets}
    mae = statistics.mean(abs(graded[it["id"]] - targets[it["label"]]) for it in items if it["id"] in graded)
    spread = [statistics.pstdev([s for s in v if s is not None]) for v in scores.values()
              if len([s for s in v if s is not None]) > 1]
    confusion = {lab: {p: 0 for p in (*targets, "ungraded")} for lab in targets}
    for it in items:
        confusion[it["label"]][label_of(firsts[it["id"]]) or "ungraded"] += 1
    metrics = {
        "n_items": len(items), "repeats": args.repeats,
        "precheck_settled": len(settled), "precheck_agree": right,
        "agreement": agree / len(items), "mean_abs_error": mae,
        "ungraded_rate": 1 - len(graded) / len(items),
        "repeatability_std": statistics.mean(spread) if spread else 0.0,
        "mean_score_correct": per_label["correct"], "mean_score_partial": per_label["partial"],
        "mean_score_wrong": per_label["wrong"],
        "llm_calls_per_pass": len(rec.calls) / args.repeats,
    }
    print(f"\nprovider={args.provider} model={args.model} mode={args.mode}  repeats={args.repeats}")
    print(f"agreement with labels      {metrics['agreement']:.0%}  ({agree}/{len(items)})")
    print(f"mean abs error vs target   {mae:.3f}   (0 = perfect)")
    print(f"mean score  correct/partial/wrong   {per_label['correct']:.2f} / {per_label['partial']:.2f} / {per_label['wrong']:.2f}")
    print(f"repeatability (mean std)   {metrics['repeatability_std']:.3f}   (0 = same score every time)")
    print(f"ungraded                   {metrics['ungraded_rate']:.0%}      LLM calls per pass {metrics['llm_calls_per_pass']:.1f}")
    print("confusion (rows = label, columns = predicted):")
    for lab, row in confusion.items():
        print(f"  {lab:<8}" + "".join(f"{p}:{n:<3} " for p, n in row.items()))
    out = Path(args.out).resolve() if args.out else ROOT / "eval" / "results" / f"{date.today()}-grading-{args.provider}-{args.mode}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"date": str(date.today()), "provider": args.provider, "model": args.model,
                               "mode": args.mode, "metrics": metrics, "confusion": confusion}, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
