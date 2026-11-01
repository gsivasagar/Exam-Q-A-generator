"""Tiny in-memory BM25 for ranking the chunks of one document set (no search server needed)."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Sequence

_STOP = frozenset("""a an and are as at be by for from has have in is it its of on or that the this to was were
what which who whom why how when where will with about into than then so such not no do does did can""".split())


def tokenize(text: str) -> List[str]:
    """Lower-case word tokens, minus stop-words, with a light plural strip ("cells" -> "cell")."""
    out = []
    for w in re.findall(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*", text.lower()):
        if len(w) < 2 or w in _STOP:
            continue
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return out


class BM25:
    def __init__(self, documents: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self._tf = [Counter(tokenize(d)) for d in documents]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg = (sum(self._len) / len(self._len)) if self._len else 0.0
        df: Counter = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(documents)
        self._idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: str) -> List[float]:
        terms = tokenize(query)
        out = []
        for tf, length in zip(self._tf, self._len):
            s = 0.0
            for t in terms:
                f = tf.get(t, 0)
                if f:
                    denom = f + self.k1 * (1 - self.b + self.b * length / (self._avg or 1.0))
                    s += self._idf.get(t, 0.0) * f * (self.k1 + 1) / denom
            out.append(s)
        return out

    def rank(self, query: str, k: int) -> List[int]:
        """Indices of the top-k documents with a positive score, best first."""
        sc = self.scores(query)
        order = sorted((i for i, s in enumerate(sc) if s > 0), key=lambda i: -sc[i])
        return order[:k]
