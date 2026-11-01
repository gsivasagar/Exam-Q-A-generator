"""Shared helpers for the evaluation scripts: provider selection, a recording wrapper, and two stand-ins
that need no model (an extractive question generator and a lexical grader).

The stand-ins make the harness runnable anywhere and give a floor to compare real models against. They
measure the PIPELINE (selection, validation, dedupe, grading plumbing), not the quality of a language model.
"""
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.bm25 import tokenize  # noqa: E402
from app.llm import get_provider  # noqa: E402


def stems(text):
    return set(tokenize(text))


# ---------------------------------------------------------------- recording wrapper
class Recorder:
    """Wraps any provider; records every prompt, reply, schema and the time each call took."""

    def __init__(self, inner):
        self.inner = inner
        self.calls = []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def generate(self, prompt, *, schema=None, temperature=None):
        t0 = time.perf_counter()
        reply = self.inner.generate(prompt, schema=schema, temperature=temperature)
        self.calls.append({"prompt": prompt, "schema": schema, "reply": reply, "seconds": time.perf_counter() - t0})
        return reply

    def usage(self):
        fn = getattr(self.inner, "usage_summary", None)
        return fn() if fn else {}


# ---------------------------------------------------------------- stand-in question generator
def question_kind(prompt, schema):
    props = ((schema or {}).get("properties", {}).get("questions", {}).get("items", {}).get("properties", {}))
    if "distractors" in props:
        return "mcq"
    if "True or False" in prompt:
        return "tf"
    if "replaced by ___" in prompt:
        return "cloze"
    return "short"


class StubGenerator:
    """Builds questions straight from the numbered passages in the prompt. Deterministic, no model."""
    max_concurrency = 1

    def __init__(self):
        self.i = 0

    def generate(self, prompt, *, schema=None, temperature=None):
        props = (schema or {}).get("properties", {})
        if "results" in props:  # verifier call: accept everything
            n = len(re.findall(r"^Question \d+:", prompt, re.M))
            return json.dumps({"results": [{"id": k, "supported": True} for k in range(1, n + 1)]})
        ctx = prompt.split("<context>")[-1].replace("</context>", "")
        passages = [(int(m.group(1)), m.group(2).strip())
                    for m in re.finditer(r"\[(\d+)\] (.+?)(?=\n\n\[\d+\] |\Z)", ctx.strip(), re.S)]
        if not passages:
            return json.dumps({"questions": []})
        sents = {n: [s for s in re.split(r"(?<=[.!?])\s+", t) if len(s.split()) >= 6] or [t] for n, t in passages}
        m = re.search(r"Generate (\d+)", prompt)
        n_req = int(m.group(1)) if m else 5
        kind = question_kind(prompt, schema)
        out = []
        for j in range(n_req):
            num, _ = passages[(self.i + j) % len(passages)]
            pool = sents[num]
            s = pool[((self.i + j) // len(passages)) % len(pool)].rstrip(".")
            words = s.split()
            item = {"topic": "Notes", "chunk": [k + 1 for k, (n, _) in enumerate(passages) if n == num][0]}
            if kind == "mcq":
                others = [x for _, t in passages for x in sents[_] if x.rstrip(".") != s][:3]
                item.update(question=f"Which statement is stated in the notes: \"{' '.join(words[:5])} ...\"?",
                            answer=s[:140], distractors=[o[:140] for o in (others + ["None", "All", "Some"])[:3]])
            elif kind == "tf":
                true = j % 2 == 0
                item.update(question=s + "." if true else s.replace(words[len(words) // 2], "nothing", 1) + ".",
                            answer="True" if true else "False")
            elif kind == "cloze":
                target = max(words, key=len)
                item.update(question=s.replace(target, "___", 1) + ".", answer=target.strip(",.;"))
            else:
                half = max(2, len(words) // 2)
                item.update(question=f"What do the notes say about \"{' '.join(words[:6])}\"?", answer=s + ".",
                            key_points=[" ".join(words[:half]), " ".join(words[half:])])
            item["explanation"] = "Stated in the passage."
            out.append(item)
        self.i += n_req
        return json.dumps({"questions": out})


# ---------------------------------------------------------------- stand-in grader
class LexicalGrader:
    """Grades by word overlap (no model). A floor: a real model should beat it on partial answers."""
    max_concurrency = 1

    def generate(self, prompt, *, schema=None, temperature=None):
        # the instructions also mention the tag, so take the LAST <student_answer> block: the real answer
        student = prompt.rsplit("<student_answer>", 1)[-1].split("</student_answer>")[0].strip()
        have = stems(student)
        if "Key points:" in prompt:
            points = re.findall(r"^\d+\. (.+)$", prompt.split("Key points:")[1].split("<student_answer>")[0], re.M)
            covered = []
            for p in points:
                want = stems(p)
                covered.append(bool(want) and len(want & have) / len(want) >= 0.6)
            return json.dumps({"covered": covered, "feedback": "lexical overlap"})
        ref = (re.search(r"Correct answer: (.+)", prompt) or [None, ""])[1]
        want = stems(ref)
        score = len(want & have) / len(want) if want else 0.0
        return json.dumps({"score": round(score, 2), "feedback": "lexical overlap"})


def make_provider(name, model=None, task="generation"):
    """('stub' | 'ollama' | 'gemini', model) -> a provider object."""
    if name == "stub":
        return StubGenerator() if task == "generation" else LexicalGrader()
    return get_provider("Gemini" if name == "gemini" else "Ollama", ollama_model=model)
