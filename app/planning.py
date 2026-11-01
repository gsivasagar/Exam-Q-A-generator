"""How many questions of each type to ask for, and when to stop."""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Set, Tuple

BATCH_SIZE = 5  # questions requested per LLM call
QUESTION_TYPES = ("short", "mcq", "mixed", "tf", "cloze")


def type_quota(question_type: str, n: int) -> Dict[str, int]:
    """Target count per concrete type; "mixed" splits roughly half and half."""
    if question_type == "mixed":
        return {"mcq": n // 2, "short": n - n // 2}
    return {question_type: n}


def max_rounds(n: int) -> int:
    return 2 * (n // BATCH_SIZE + 1) + 2


def next_request(quota: Dict[str, int], pairs: List[Dict], failed: Set[str]) -> Optional[Tuple[str, int]]:
    """The (type, shortfall) furthest from its quota, skipping types whose batches already came back empty.
    None when every remaining type is satisfied or has failed."""
    deficit = {}
    for t, target in quota.items():
        missing = target - sum(p["type"] == t for p in pairs)
        if missing > 0 and t not in failed:
            deficit[t] = missing
    if not deficit:
        return None
    qtype = max(deficit, key=deficit.get)
    return qtype, deficit[qtype]


def trim_to_quota(pairs: List[Dict], quota: Dict[str, int], question_type: str) -> List[Dict]:
    final: List[Dict] = []
    for t, q in quota.items():
        final += [p for p in pairs if p["type"] == t][:q]
    if question_type == "mixed":
        random.shuffle(final)
    return final
