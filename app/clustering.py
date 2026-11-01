"""Small, dependency-light k-means (cosine) for grouping chunk embeddings."""
from __future__ import annotations

from typing import List

import numpy as np


def _normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms == 0, 1.0, norms)


def kmeans(vectors, k: int, *, iters: int = 30, seed: int = 0) -> List[int]:
    """Cluster `vectors` into at most `k` groups by cosine distance; returns one label per row.

    k-means++ initialisation with a fixed seed, so the same input always gives the same clusters."""
    x = _normalize(np.asarray(vectors, dtype=np.float64))
    n = len(x)
    k = max(1, min(k, n))
    rng = np.random.RandomState(seed)

    centers = [x[rng.randint(n)]]
    for _ in range(1, k):
        d2 = np.maximum(0.0, 2.0 - 2.0 * (x @ np.array(centers).T)).min(axis=1)
        total = d2.sum()
        idx = rng.randint(n) if total <= 1e-12 else rng.choice(n, p=d2 / total)
        centers.append(x[idx])
    c = np.array(centers)

    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        labels = (x @ c.T).argmax(axis=1)
        new = c.copy()
        for j in range(k):
            members = x[labels == j]
            if len(members):
                new[j] = members.mean(axis=0)
        new = _normalize(new)
        if np.allclose(new, c):
            break
        c = new
    return [int(v) for v in labels]


def mean_pairwise_similarity(vectors) -> float:
    """Average cosine similarity between distinct rows (lower = more diverse). 0.0 for fewer than 2 rows."""
    x = _normalize(np.asarray(vectors, dtype=np.float64))
    n = len(x)
    if n < 2:
        return 0.0
    sims = x @ x.T
    return float((sims.sum() - np.trace(sims)) / (n * (n - 1)))


def farthest_first_order(centroids) -> List[int]:
    """Order points so each next one is as far (cosine) as possible from all those already chosen.
    Starts from index 0. Used to rotate through clusters so neighbours in the order are dissimilar."""
    x = _normalize(np.asarray(centroids, dtype=np.float64))
    n = len(x)
    order = [0]
    nearest = x @ x[0]  # similarity to the closest chosen point
    chosen = {0}
    while len(order) < n:
        cand = [i for i in range(n) if i not in chosen]
        nxt = min(cand, key=lambda i: nearest[i])
        order.append(nxt)
        chosen.add(nxt)
        nearest = np.maximum(nearest, x @ x[nxt])
    return order
