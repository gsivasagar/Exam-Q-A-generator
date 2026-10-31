from collections import Counter
from typing import Dict, List

from . import config


def recommend(graded: List[Dict], limit: int = 3) -> List[str]:
    """Return up to `limit` topics where the student scored below the pass mark."""
    weak = [g.get("topic", "General") for g in graded
            if g.get("score") is not None and g["score"] < config.PASS_THRESHOLD]
    return [t for t, _ in Counter(weak).most_common(limit)]
