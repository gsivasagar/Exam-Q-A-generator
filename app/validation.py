"""Parse and validate LLM question output."""
from __future__ import annotations

import json
import random
import re
from typing import Dict, List, Optional, Sequence

N_OPTIONS = 4


def extract_json_array(text: str) -> list:
    """Pull a JSON array out of an LLM reply (tolerates prose, code fences, a
    top-level object wrapping the list, curly quotes and trailing commas)."""
    text = text.replace("“", '"').replace("”", '"')
    text = re.sub(r"```(?:json)?", "", text)
    candidates = []
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])
    for cand in candidates:
        cand = re.sub(r",\s*([\]}])", r"\1", cand)  # trailing commas
        try:
            parsed = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, list):
                    return v
            if "question" in parsed:  # model returned a single bare question object
                return [parsed]
    raise ValueError("No valid JSON array found in model response.")


def match_option(answer: str, options: List[str]) -> Optional[str]:
    """Return the option the answer refers to (by text, or by letter 'B' / 'B)')."""
    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip().lower()

    for o in options:
        if norm(o) == norm(answer):
            return o
    m = re.fullmatch(r"\(?([A-Da-d])[\).:]?", answer.strip())
    if m and ord(m.group(1).upper()) - 65 < len(options):
        return options[ord(m.group(1).upper()) - 65]
    return None


def distinct(values) -> List[str]:
    out, seen = [], set()
    for v in values:
        v = str(v).strip()
        if v and v.lower() not in seen:
            seen.add(v.lower())
            out.append(v)
    return out


def build_options(item: dict, answer: str) -> Optional[List[str]]:
    """Return N_OPTIONS distinct options that include the answer, or None.

    Preferred input is `distractors` (wrong answers only): options are built in
    code, so the model can't forget the correct one. A full `options` list that
    contains the answer is accepted too."""
    d = item.get("distractors")
    if isinstance(d, list):
        wrong = [x for x in distinct(d) if x.lower() != answer.lower()]
        if len(wrong) >= N_OPTIONS - 1:
            return [answer] + wrong[:N_OPTIONS - 1]
        return None
    o = item.get("options")
    if isinstance(o, list):
        options = distinct(o)
        if len(options) == N_OPTIONS and match_option(answer, options) is not None:
            return options
    return None


TF_ANSWERS = {"true": "True", "t": "True", "yes": "True", "false": "False", "f": "False", "no": "False"}
_BLANK = re.compile(r"_{2,}|\[blank\]", re.I)


def validate_pairs(items: list, hits: Optional[Sequence[dict]] = None, qtype: str = "short",
                   keep_chunk: bool = False) -> List[Dict]:
    """Keep only well-formed items. For MCQ: N_OPTIONS distinct options (answer + distractors, shuffled).
    "tf": a statement answered True/False. "cloze": a sentence with a blank. `chunk` is mapped back to
    source/page; with keep_chunk the 0-based index is kept as `_chunk` (used by the verifier, then dropped)."""
    good = []
    for it in items:
        if not isinstance(it, dict):
            continue
        q = str(it.get("question", "")).strip()
        a = str(it.get("answer", "")).strip()
        if not q or not a:
            continue
        pair = {"question": q, "answer": a, "type": "short",
                "topic": str(it.get("topic") or "General").strip() or "General",
                "source": None, "page": None}
        expl = str(it.get("explanation") or "").strip()
        if expl:
            pair["explanation"] = expl
        kps = it.get("key_points")
        if qtype == "short" and isinstance(kps, list):
            points = distinct(kps)[:4]
            if len(points) >= 2:
                pair["key_points"] = points
        if qtype == "tf":
            tf = TF_ANSWERS.get(a.strip().lower().rstrip("."))
            if tf is None:
                continue
            pair.update(type="tf", options=["True", "False"], answer=tf)
        elif qtype == "cloze":
            if not _BLANK.search(q):
                continue
            pair["type"] = "cloze"
        elif qtype == "mcq":
            options = build_options(it, a)
            if options is None:
                continue
            correct = match_option(a, options)
            random.shuffle(options)
            pair.update(type="mcq", options=options, answer=correct)
        try:
            idx = int(it.get("chunk")) - 1
        except (TypeError, ValueError):
            idx = -1
        if hits and 0 <= idx < len(hits):
            pair["source"], pair["page"] = hits[idx].get("source"), hits[idx].get("page")
        if keep_chunk:
            pair["_chunk"] = idx
        good.append(pair)
    return good
