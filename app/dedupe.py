"""Remove duplicate questions."""
from __future__ import annotations

import re
from typing import Dict, List


def dedupe(pairs: List[Dict]) -> List[Dict]:
    """Keep the first of any questions that are identical after lower-casing and stripping punctuation."""
    seen, out = set(), []
    for p in pairs:
        key = re.sub(r"\W+", " ", p["question"].lower()).strip()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def semantic_dedupe(pairs: List[Dict], embed, threshold: float = 0.9) -> List[Dict]:
    """Drop later questions whose embedding is within `threshold` cosine similarity of an earlier one
    (paraphrases the text-based `dedupe` cannot see). `embed(list[str]) -> list[vector]`."""
    if len(pairs) < 2:
        return list(pairs)
    import numpy as np

    vecs = np.asarray(embed([p["question"] for p in pairs]), dtype=np.float64)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.where(norms == 0, 1.0, norms)
    kept: List[int] = []
    for i in range(len(pairs)):
        if all(float(vecs[i] @ vecs[j]) < threshold for j in kept):
            kept.append(i)
    return [pairs[i] for i in kept]
