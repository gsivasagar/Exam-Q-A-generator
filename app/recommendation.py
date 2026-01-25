from typing import List, Dict
from collections import Counter

def recommend(graded: List[Dict]) -> List[str]:
    """Return top 3 topics where the user scored <0.6."""
    weak = [item["topic"] for item in graded if item["score"] < 0.6]
    counts = Counter(weak)
    return [t for t, _ in counts.most_common(3)]