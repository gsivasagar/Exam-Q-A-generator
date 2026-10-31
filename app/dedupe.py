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
