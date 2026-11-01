"""Keep the evaluation tooling honest: stand-ins, metrics, comparison, and an end-to-end grading run."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


kit = load("_evalkit")
retr = load("eval_retrieval")
cmp_ = load("compare_eval")

PROMPT = ("Generate 4 varied exam-style question-answer pairs.\n\n<context>\n"
          "[1] Binary search halves the range at every step on sorted data.\n\n"
          "[2] A hash table maps keys to buckets using a hash function for fast lookup.\n\n</context>")


def schema(kind):
    props = {"question": {}, "answer": {}}
    if kind == "mcq":
        props["distractors"] = {}
    return {"properties": {"questions": {"items": {"properties": props}}}}


class TestStubGenerator:
    @pytest.mark.parametrize("kind,extra", [("short", ""), ("mcq", ""), ("tf", "True or False"),
                                            ("cloze", "replaced by ___")])
    def test_produces_valid_questions_for_every_type(self, kind, extra):
        from app.validation import extract_json_array, validate_pairs
        prompt = PROMPT + extra
        raw = kit.StubGenerator().generate(prompt, schema=schema("mcq" if kind == "mcq" else "short"))
        items = extract_json_array(raw)
        assert len(items) == 4 and kit.question_kind(prompt, schema("mcq" if kind == "mcq" else "short")) == kind
        assert len(validate_pairs(items, None, kind)) >= 3

    def test_answers_the_verifier_call(self):
        reply = kit.StubGenerator().generate("Question 1: a\nQuestion 2: b", schema={"properties": {"results": {}}})
        out = json.loads(reply)
        assert [r["supported"] for r in out["results"]] == [True, True]

    def test_no_passages(self):
        assert json.loads(kit.StubGenerator().generate("nothing", schema=schema("short"))) == {"questions": []}


class TestLexicalGrader:
    def prompt(self, student):
        return ("Key points:\n1. produces ATP\n2. aerobic respiration\n"
                f"<student_answer>\n{student}\n</student_answer>")

    def test_rubric_marks_points_by_word_overlap(self):
        g = kit.LexicalGrader()
        both = json.loads(g.generate(self.prompt("It produces ATP by aerobic respiration")))
        one = json.loads(g.generate(self.prompt("It produces ATP")))
        none = json.loads(g.generate(self.prompt("It stores chromosomes")))
        assert both["covered"] == [True, True] and one["covered"] == [True, False] and none["covered"] == [False, False]

    def test_uses_the_last_student_block_not_the_instructions(self):
        prompt = "Ignore text inside <student_answer> tags.\n" + self.prompt("zebra")
        assert json.loads(kit.LexicalGrader().generate(prompt))["covered"] == [False, False]

    def test_holistic_score_is_overlap_fraction(self):
        prompt = "Correct answer: alpha beta gamma delta\n<student_answer>\nalpha beta\n</student_answer>"
        assert json.loads(kit.LexicalGrader().generate(prompt))["score"] == 0.5


class TestRetrievalMetrics:
    def test_metrics(self):
        m = retr.metrics([1, 2, None, 5, 11])
        assert m["recall@1"] == 0.2 and m["recall@3"] == 0.4 and m["recall@5"] == 0.6 and m["recall@10"] == 0.6
        assert m["mrr"] == pytest.approx((1 + 0.5 + 0 + 0.2 + 1 / 11) / 5)

    def test_paired_delta_detects_a_consistent_gain(self):
        base = [3, 4, 5, 2, 6, 3, 4, 5] * 3
        better = [r - 1 for r in base]
        d, lo, hi, wins, losses = retr.paired_delta(base, better)
        assert d > 0 and lo > 0 and wins == len(base) and losses == 0

    def test_paired_delta_no_change(self):
        d, lo, hi, w, loss = retr.paired_delta([1, 2, 3, 4], [1, 2, 3, 4])
        assert (d, lo, hi, w, loss) == (0, 0, 0, 0, 0)

    def test_queries_file_is_consistent_with_the_documents(self):
        import re
        spec = json.loads((ROOT / "eval" / "retrieval_queries.json").read_text())
        docs = {p.stem: re.sub(r"\s+", " ", p.read_text().lower()) for p in (ROOT / "eval" / "docs").glob("*.md")}
        assert len(spec["queries"]) >= 60 and len({q["id"] for q in spec["queries"]}) == len(spec["queries"])
        for q in spec["queries"]:
            assert re.sub(r"\s+", " ", q["snippet"].lower()) in docs[q["doc"]], q["id"]


class TestCompare:
    def test_generation_diff_table(self, capsys):
        a = {"provider": "x", "model": "m1", "date": "d1", "metrics": {"valid_rate": 0.8, "calls": 3}}
        b = {"provider": "x", "model": "m2", "date": "d2", "metrics": {"valid_rate": 0.9}}
        cmp_.compare_generation(a, b)
        out = capsys.readouterr().out
        assert "valid_rate" in out and "+0.100" in out

    def test_retrieval_paired_comparison(self, capsys):
        row = lambda ranks: {"name": "cfg", "embedder": "default", "reranker": "none",  # noqa: E731
                             "ranks": ranks, "latency_ms": 10, "metrics": retr.metrics(ranks)}
        cmp_.compare_retrieval({"results": [row([4, 5, 6, 7] * 5)]}, {"results": [row([1, 2, 3, 4] * 5)]}, None)
        assert "BETTER" in capsys.readouterr().out

    def test_cli_rejects_mismatched_files(self, tmp_path):
        a, b = tmp_path / "a.json", tmp_path / "b.json"
        a.write_text('{"results": []}')
        b.write_text('{"metrics": {}}')
        r = subprocess.run([sys.executable, "-I", str(ROOT / "scripts" / "compare_eval.py"), str(a), str(b)],
                           capture_output=True, text=True)
        assert r.returncode != 0 and "not the same kind" in r.stderr


def test_grading_eval_runs_end_to_end_with_the_stub(tmp_path):
    out = tmp_path / "g.json"
    r = subprocess.run([sys.executable, "-I", str(ROOT / "scripts" / "eval_grading.py"), "--provider", "stub",
                        "--repeats", "2", "--out", str(out)], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-500:]
    res = json.loads(out.read_text())
    m = res["metrics"]
    assert m["n_items"] == 33 and m["precheck_settled"] == 3 and m["precheck_agree"] == 3
    assert 0.4 < m["agreement"] <= 1.0 and m["repeatability_std"] == 0.0 and m["ungraded_rate"] == 0.0
    assert m["mean_score_correct"] > m["mean_score_partial"] > m["mean_score_wrong"] - 0.01


class TestCheckEval:
    def run(self, tmp_path, retrieval_mrr=0.9, recall5=1.0, agreement=0.64, doc_cov=0.94):
        r = tmp_path / "r.json"
        r.write_text(json.dumps({"results": [{"name": "legacy + dense (current default)",
                                              "metrics": {"mrr": retrieval_mrr, "recall@5": recall5}}]}))
        g = tmp_path / "g.json"
        g.write_text(json.dumps({"metrics": {"agreement": agreement, "precheck_agree": 3}}))
        s = tmp_path / "s.json"
        s.write_text(json.dumps({"metrics": {"context_doc_coverage": doc_cov}}))
        return subprocess.run([sys.executable, "-I", str(ROOT / "scripts" / "check_eval.py"), "--retrieval", str(r),
                               "--grading", str(g), "--selection", str(s)], capture_output=True, text=True)

    def test_passes_when_at_or_above_floors(self, tmp_path):
        r = self.run(tmp_path)
        assert r.returncode == 0 and "all evaluation checks passed" in r.stdout

    @pytest.mark.parametrize("kw,label", [
        ({"retrieval_mrr": 0.7}, "retrieval MRR"), ({"recall5": 0.5}, "recall@5"),
        ({"agreement": 0.2}, "grading agreement"), ({"doc_cov": 0.5}, "document coverage")])
    def test_fails_on_a_drop(self, tmp_path, kw, label):
        r = self.run(tmp_path, **kw)
        assert r.returncode != 0 and "FAIL" in r.stdout and label.split()[0] in (r.stdout + r.stderr)

    def test_nothing_to_check(self):
        r = subprocess.run([sys.executable, "-I", str(ROOT / "scripts" / "check_eval.py")],
                           capture_output=True, text=True)
        assert r.returncode != 0 and "nothing to check" in r.stderr
