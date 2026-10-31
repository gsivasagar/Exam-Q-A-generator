"""Multiple choice, citations, mixed quizzes and multi-topic (weak-topic) quizzes."""
import json

import pytest

from app.grader import grade_mcq, grade_question
from app.qa_generator import _match_option, generate_qa_pairs, ingest_pdf, validate_pairs
from tests.conftest import FakeLLM, make_pdf_bytes

HITS = [{"text": "t1", "source": "a.pdf", "page": 3}, {"text": "t2", "source": "b.pdf", "page": 7}]


def mcq(**over):
    base = {"question": "Which?", "options": ["w", "x", "y", "z"], "answer": "x", "topic": "T", "chunk": 1}
    base.update(over)
    return base


class TestMatchOption:
    @pytest.mark.parametrize("ans", ["x", " X ", "B", "b)", "(B)", "B."])
    def test_matches(self, ans):
        assert _match_option(ans, ["w", "x", "y", "z"]) == "x"

    @pytest.mark.parametrize("ans", ["nope", "E", ""])
    def test_no_match(self, ans):
        assert _match_option(ans, ["w", "x", "y", "z"]) is None


class TestValidateMcq:
    def test_valid_mcq_keeps_answer_among_shuffled_options(self):
        out = validate_pairs([mcq()], HITS, "mcq")
        assert out[0]["type"] == "mcq" and out[0]["answer"] in out[0]["options"]
        assert sorted(out[0]["options"]) == ["w", "x", "y", "z"]

    def test_letter_answer_resolved_to_text(self):
        assert validate_pairs([mcq(answer="C")], HITS, "mcq")[0]["answer"] == "y"

    @pytest.mark.parametrize("bad", [
        mcq(options=["w", "x", "y"]),                 # too few
        mcq(options=["w", "x", "y", "z", "q"]),       # too many
        mcq(options=["w", "w", "y", "z"]),            # duplicates -> only 3 distinct
        mcq(answer="not an option"),                  # answer not in options
        mcq(options="w x y z"),                       # wrong type
    ])
    def test_rejects_malformed(self, bad):
        assert validate_pairs([bad], HITS, "mcq") == []

    def test_distractors_build_options_in_code(self):
        item = {"question": "Where?", "answer": "stroma", "topic": "T", "chunk": 1,
                "distractors": ["cytoplasm", "ribosome", "nucleus"]}
        out = validate_pairs([item], HITS, "mcq")[0]
        assert sorted(out["options"]) == ["cytoplasm", "nucleus", "ribosome", "stroma"]
        assert out["answer"] == "stroma"

    def test_regression_model_that_lists_only_wrong_options_is_rejected_not_shown(self):
        # real gemma3 output: 4 options, none of them the answer. Must never reach a student.
        item = {"question": "Where?", "answer": "stroma", "topic": "T", "chunk": 1,
                "options": ["mitochondria", "cytoplasm", "ribosome", "chloroplast"]}
        assert validate_pairs([item], HITS, "mcq") == []

    @pytest.mark.parametrize("d", [["a", "b"], ["a", "a", "a"], ["x", "x", "b"], "a,b,c", []])
    def test_too_few_or_duplicate_distractors_rejected(self, d):
        item = {"question": "q", "answer": "x", "topic": "T", "distractors": d}
        assert validate_pairs([item], None, "mcq") == []

    def test_extra_distractors_trimmed_and_answer_removed_from_them(self):
        item = {"question": "q", "answer": "X", "topic": "T", "distractors": ["x", "a", "b", "c", "d"]}
        out = validate_pairs([item], None, "mcq")[0]
        assert sorted(o.lower() for o in out["options"]) == ["a", "b", "c", "x"]

    def test_missing_options_rejected_for_mcq_but_fine_for_short(self):
        item = {"question": "q", "answer": "a", "topic": "T"}
        assert validate_pairs([item], None, "mcq") == []
        assert validate_pairs([item], None, "short")[0]["type"] == "short"


class TestCitations:
    def test_chunk_number_maps_to_source_and_page(self):
        out = validate_pairs([mcq(chunk=2)], HITS, "mcq")
        assert (out[0]["source"], out[0]["page"]) == ("b.pdf", 7)

    @pytest.mark.parametrize("chunk", [0, 3, -1, "abc", None, 99])
    def test_bad_chunk_number_means_no_citation_not_crash(self, chunk):
        out = validate_pairs([mcq(chunk=chunk)], HITS, "mcq")
        assert out and out[0]["source"] is None and out[0]["page"] is None

    def test_end_to_end_citation_points_at_real_page(self, store):
        from fpdf import FPDF
        pdf = FPDF()
        for marker in ("ALPHA", "BETA"):
            pdf.add_page()
            pdf.set_font("Helvetica", size=11)
            pdf.multi_cell(0, 6, " ".join(f"{marker} fact {i}." for i in range(20)))
        ingest_pdf(None, "c", filename="n.pdf", data=bytes(pdf.output()))

        class Cite(FakeLLM):
            def generate(self, prompt, *, schema=None):
                # cite whichever passage mentions BETA
                import re
                n = int(re.search(r"\[(\d+)\] BETA", prompt).group(1)) if "] BETA" in prompt else 1
                return json.dumps({"questions": [{"question": f"q{self.calls}?", "answer": "a",
                                                  "topic": "T", "chunk": n}]})
        pairs = generate_qa_pairs("c", Cite(), n=1)
        assert pairs[0]["source"] == "n.pdf"


class TestGradeMcq:
    QA = {"type": "mcq", "answer": "Stroma", "options": ["Stroma", "Cytoplasm", "Nucleus", "Matrix"]}

    def test_correct(self):
        assert grade_mcq(self.QA, "stroma")["score"] == 1.0

    def test_wrong_reveals_answer(self):
        r = grade_mcq(self.QA, "Nucleus")
        assert r["score"] == 0.0 and "Stroma" in r["feedback"]

    def test_unanswered(self):
        assert grade_mcq(self.QA, "")["score"] == 0.0

    def test_mcq_never_calls_llm(self):
        llm = FakeLLM()
        grade_question(self.QA, "Stroma", llm)
        assert llm.calls == 0

    def test_short_dispatches_to_llm_and_blank_skips_it(self):
        llm = FakeLLM(['{"score": 0.5, "feedback": "meh"}'])
        assert grade_question({"type": "short", "question": "q", "answer": "a"}, "stu", llm)["score"] == 0.5
        assert grade_question({"type": "short", "question": "q", "answer": "a"}, "  ", llm)["score"] == 0.0
        assert llm.calls == 1


class TestGenerationModes:
    def _ingest(self, doc="m"):
        ingest_pdf(None, doc, filename="a.pdf", data=make_pdf_bytes(
            [f"Mitochondria produce energy number {i} for the cell." for i in range(60)]))

    class Typed(FakeLLM):
        """Answers according to the schema it is asked for (mcq vs short)."""
        def generate(self, prompt, *, schema=None):
            self.calls += 1
            self.prompts.append(prompt)
            is_mcq = "distractors" in schema["properties"]["questions"]["items"]["properties"]
            qs = []
            for i in range(5):
                q = {"question": f"{'M' if is_mcq else 'S'}{self.calls}-{i}?", "answer": "x",
                     "topic": "T", "chunk": 1}
                if is_mcq:
                    q.update(distractors=["w", "y", "z"])
                qs.append(q)
            return json.dumps({"questions": qs})

    def test_mcq_mode(self, store):
        self._ingest()
        pairs = generate_qa_pairs("m", self.Typed(), n=7, question_type="mcq")
        assert len(pairs) == 7 and all(p["type"] == "mcq" and len(p["options"]) == 4 for p in pairs)

    def test_mixed_mode_contains_both(self, store):
        self._ingest()
        pairs = generate_qa_pairs("m", self.Typed(), n=10, question_type="mixed")
        assert {p["type"] for p in pairs} == {"short", "mcq"}

    @pytest.mark.parametrize("n", [5, 6, 10, 11])
    def test_mixed_is_balanced_and_exact(self, store, n):
        self._ingest()
        pairs = generate_qa_pairs("m", self.Typed(), n=n, question_type="mixed")
        assert len(pairs) == n
        assert sum(p["type"] == "mcq" for p in pairs) == n // 2
        assert sum(p["type"] == "short" for p in pairs) == n - n // 2

    def test_mixed_survives_one_type_failing(self, store):
        self._ingest()
        class OnlyShort(self.Typed):
            def generate(self, prompt, *, schema=None):
                if "distractors" in schema["properties"]["questions"]["items"]["properties"]:
                    self.calls += 1
                    return "garbage"
                return super().generate(prompt, schema=schema)
        pairs = generate_qa_pairs("m", OnlyShort(), n=6, question_type="mixed")
        assert pairs and all(p["type"] == "short" for p in pairs)   # degrades, doesn't crash

    def test_invalid_type_rejected(self, store):
        with pytest.raises(ValueError):
            generate_qa_pairs("m", FakeLLM(), question_type="essay")

    def test_mcq_prompt_demands_options_and_numbered_passages(self, store):
        self._ingest()
        llm = self.Typed()
        generate_qa_pairs("m", llm, n=5, question_type="mcq")
        assert "distractors" in llm.prompts[0] and "[1]" in llm.prompts[0]

    def test_multi_topic_list_for_weak_topic_quiz(self, store):
        self._ingest()
        llm = self.Typed()
        pairs = generate_qa_pairs("m", llm, n=5, topic=["Glycolysis", "Calvin cycle"])
        assert pairs and "Glycolysis, Calvin cycle" in llm.prompts[0]

    def test_empty_topic_list_means_whole_document(self, store):
        self._ingest()
        assert generate_qa_pairs("m", self.Typed(), n=5, topic=[])
