"""Question-generation evaluation over eval/docs.

Runs the real pipeline (routing, selection, prompting, validation, dedupe, optional verifier) once per
document and once over all documents combined, `--runs` times, and reports:

  valid_rate         usable questions / requested
  parse_fail_rate    model replies that were not parseable JSON
  item_validity      well-formed items / items the model wrote (MCQ: 4 distinct options etc.)
  grounded_rate      questions whose answer words are found in the source document (a proxy for citation
                     correctness; the file also writes a sample for manual checking)
  section_coverage   distinct document sections touched / sections available (per-document quizzes)
  doc_coverage       distinct documents touched / documents available (combined quiz)
  duplicate_rate     questions that nearly repeat one from an earlier run on the same document
  llm_calls / seconds per quiz, and tokens when the provider reports them

  python scripts/eval_generation.py --provider stub                          # pipeline only, no model
  python scripts/eval_generation.py --provider ollama --model gemma3 --types short,mcq --runs 3
  python scripts/eval_generation.py --provider gemini --verify --out eval/results/gemini-verify.json
Compare two result files with scripts/compare_eval.py."""
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

import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _evalkit import ROOT, Recorder, make_provider, question_kind, stems  # noqa: E402

from app import config, vector_store  # noqa: E402
from app.embeddings import get_embedding_function  # noqa: E402
from app.qa_generator import generate_qa_pairs, ingest_document  # noqa: E402
from app.validation import extract_json_array, validate_pairs  # noqa: E402


def sections_of(text):
    """{heading: body} for a markdown document with '## ' sections."""
    parts = re.split(r"^## (.+)$", text, flags=re.M)
    return {parts[i].strip(): parts[i + 1] for i in range(1, len(parts) - 1, 2)}


def best_section(secs, q):
    want = stems(q["answer"] + " " + q["question"])
    return max(secs, key=lambda h: len(want & stems(secs[h]))) if want else None


def grounded(doc_text, q):
    want = stems(q["answer"])
    if q.get("type") == "tf":
        want = stems(q["question"])
    return bool(want) and len(want & stems(doc_text)) / len(want) >= 0.7


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", choices=["stub", "ollama", "gemini"], default="stub")
    ap.add_argument("--model", default=None)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--n", type=int, default=8, help="questions requested per quiz")
    ap.add_argument("--types", default="short,mcq", help="comma list of short,mcq,tf,cloze,mixed")
    ap.add_argument("--selection", choices=["coverage", "random"], default=config.SELECTION_STRATEGY)
    ap.add_argument("--routing", choices=["on", "off"], default="on" if config.ADAPTIVE_ROUTING else "off")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--scope", choices=["both", "each", "all"], default="both",
                    help="each = one quiz per document, all = one quiz over all documents combined")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    docs = {p.stem: p for p in sorted((ROOT / "eval" / "docs").glob("*.md"))}
    texts = {k: p.read_text() for k, p in docs.items()}
    secs = {k: sections_of(t) for k, t in texts.items()}
    tmp = tempfile.mkdtemp(prefix="eval_gen_")
    config.CHROMA_DIR = tmp
    config.SELECTION_STRATEGY, config.ADAPTIVE_ROUTING = args.selection, args.routing == "on"
    config.VERIFY_QUESTIONS = args.verify
    vector_store._collection.cache_clear()
    embed = get_embedding_function("default")

    agg = {"requested": 0, "usable": 0, "replies": 0, "parse_fail": 0, "items": 0, "valid_items": 0,
           "questions": 0, "grounded": 0, "calls": 0, "quizzes": 0, "seconds": 0.0}
    cov_sections, cov_docs, dup_num, dup_den, errors = [], [], 0, 0, []
    sample = []
    rng = random.Random(args.seed)
    try:
        for stem, path in docs.items():
            ingest_document(None, f"d-{stem}", filename=path.name, data=path.read_bytes())
            ingest_document(None, "all", filename=path.name, data=path.read_bytes())
        usage_total = {}
        norm_docs = {k: re.sub(r"\s+", " ", re.sub(r"^#{1,6}\s*", "", t, flags=re.M).lower())  # as the loader strips '#'
                     for k, t in texts.items()}
        ctx_docs, ctx_sims = [], []
        for qtype in [t.strip() for t in args.types.split(",") if t.strip()]:
            earlier = {k: [] for k in docs}
            for run in range(args.runs):
                random.seed(args.seed * 1000 + run)
                for scope in [*(docs if args.scope != "all" else []), *(["all"] if args.scope != "each" else [])]:
                    rec = Recorder(make_provider(args.provider, args.model))
                    t0 = time.perf_counter()
                    try:
                        qs = generate_qa_pairs(f"d-{scope}" if scope != "all" else "all", rec, n=args.n, question_type=qtype)
                    except Exception as exc:  # a model that returns nothing usable is a data point, not a crash
                        errors.append(f"{qtype}/{scope}/run{run}: {exc}")
                        qs = []
                    agg["seconds"] += time.perf_counter() - t0
                    agg["quizzes"] += 1
                    agg["requested"] += args.n
                    agg["usable"] += len(qs)
                    agg["calls"] += len(rec.calls)
                    for k, v in rec.usage().items():
                        usage_total[k] = usage_total.get(k, 0) + v
                    for c in rec.calls:
                        if "results" in ((c["schema"] or {}).get("properties", {})):
                            continue  # verifier replies are not questions
                        agg["replies"] += 1
                        try:
                            items = extract_json_array(c["reply"])
                        except ValueError:
                            agg["parse_fail"] += 1
                            continue
                        agg["items"] += len(items)
                        agg["valid_items"] += len(validate_pairs(items, None, question_kind(c["prompt"], c["schema"])))
                    for q in qs:
                        agg["questions"] += 1
                        src = q.get("source") or ""
                        text = texts.get(Path(src).stem, "")
                        agg["grounded"] += grounded(text, q)
                        if len(sample) < 400:
                            sample.append((qtype, scope, q, Path(src).stem))
                    if scope == "all":
                        cov_docs.append(len({Path(q.get("source") or "").stem for q in qs}) / len(docs))
                        # selection quality, independent of what the model wrote: the chunks it was shown
                        shown = []
                        for c in rec.calls:
                            if "results" in ((c["schema"] or {}).get("properties", {})):
                                continue
                            ctx = c["prompt"].split("<context>")[-1].replace("</context>", "")
                            shown += [re.sub(r"\s+", " ", m.group(1).strip().lower())
                                      for m in re.finditer(r"\[\d+\] (.+?)(?=\n\n\[\d+\] |\Z)", ctx.strip(), re.S)]
                        shown = list(dict.fromkeys(shown))
                        if len(shown) > 1:
                            ctx_docs.append(len({k for s in shown for k, t in norm_docs.items() if s[:50] in t}) / len(docs))
                            v = np.array(embed(shown), dtype=float)
                            v /= np.linalg.norm(v, axis=1, keepdims=True)
                            sim = v @ v.T
                            ctx_sims.append(float((sim.sum() - len(v)) / (len(v) * (len(v) - 1))))
                    else:
                        hit = {best_section(secs[scope], q) for q in qs}
                        cov_sections.append(len(hit - {None}) / max(1, len(secs[scope])))
                        # near-duplicates of questions from earlier runs on the same document
                        if earlier[scope] and qs:
                            old = np.array(embed([x["question"] for x in earlier[scope]]), dtype=float)
                            new = np.array(embed([x["question"] for x in qs]), dtype=float)
                            old /= np.linalg.norm(old, axis=1, keepdims=True)
                            new /= np.linalg.norm(new, axis=1, keepdims=True)
                            dup_num += int((new @ old.T).max(axis=1).__ge__(0.85).sum())
                            dup_den += len(qs)
                        earlier[scope] += qs
        pct = lambda a, b: (a / b) if b else 0.0  # noqa: E731
        metrics = {
            "valid_rate": pct(agg["usable"], agg["requested"]),
            "parse_fail_rate": pct(agg["parse_fail"], agg["replies"]),
            "item_validity": pct(agg["valid_items"], agg["items"]),
            "grounded_rate": pct(agg["grounded"], agg["questions"]),
            "section_coverage": statistics.mean(cov_sections) if cov_sections else 0.0,
            "doc_coverage": statistics.mean(cov_docs) if cov_docs else 0.0,
            "context_doc_coverage": statistics.mean(ctx_docs) if ctx_docs else 0.0,
            "context_similarity": statistics.mean(ctx_sims) if ctx_sims else 0.0,
            "duplicate_rate": pct(dup_num, dup_den),
            "llm_calls_per_quiz": pct(agg["calls"], agg["quizzes"]),
            "seconds_per_quiz": pct(agg["seconds"], agg["quizzes"]),
            "quizzes": agg["quizzes"], "failed_quizzes": len(errors),
        }
        for k, v in usage_total.items():
            if k != "calls":
                metrics[f"{k}_per_quiz"] = pct(v, agg["quizzes"])
        print(f"provider={args.provider} model={args.model} types={args.types} runs={args.runs} n={args.n} "
              f"selection={args.selection} routing={args.routing} verify={args.verify}")
        for k, v in metrics.items():
            print(f"  {k:<22}{v:>10.3f}" if isinstance(v, float) else f"  {k:<22}{v:>10}")
        for e in errors[:5]:
            print("  error:", e)
        out = Path(args.out).resolve() if args.out else ROOT / "eval" / "results" / (
            f"{date.today()}-generation-{args.provider}-{args.selection}-routing{args.routing}{'-verify' if args.verify else ''}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"date": str(date.today()), "provider": args.provider, "model": args.model,
                                   "config": vars(args), "metrics": metrics, "errors": errors[:20],
                                   "raw": {"context_doc_coverage": ctx_docs, "context_similarity": ctx_sims}}, indent=1))
        # a sample for manual citation / quality review
        rng.shuffle(sample)
        review = out.with_suffix(".review.md")
        lines = ["# Manual review sample", "", "For each item check: is the question answerable from the cited document, "
                 "and is the answer right? Mark OK / WRONG / UNSUPPORTED.", ""]
        for i, (qt, scope, q, stem) in enumerate(sample[:30], 1):
            ans = q["answer"]
            sec = best_section(secs.get(stem, {}), q) if stem in secs else None
            lines += [f"## {i}. [{qt}] {q['question']}", f"- answer: {ans}", f"- cited source: {q.get('source')} "
                      f"(best-matching section: {sec})", "- verdict: ", ""]
        review.write_text("\n".join(lines))
        print(f"\nwrote {out}\nwrote {review}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        vector_store._collection.cache_clear()


if __name__ == "__main__":
    main()
