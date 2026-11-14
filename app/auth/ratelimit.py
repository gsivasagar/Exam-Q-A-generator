"""A small sliding-window rate limiter kept in memory.

Correct for one server process. With more than one process or machine, move the counters to a shared
store such as Redis, or each process allows its own quota.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Callable, Deque, Dict


class RateLimiter:
    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, limit: int, window_seconds: float) -> bool:
        """True and count the attempt if fewer than `limit` happened in the last `window_seconds`."""
        now = self._clock()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] >= window_seconds:
                hits.popleft()
            if len(hits) >= limit:
                return False
            hits.append(now)
            return True

    def retry_after(self, key: str, window_seconds: float) -> int:
        """Seconds until the oldest counted attempt leaves the window (at least 1)."""
        now = self._clock()
        with self._lock:
            hits = self._hits.get(key)
            if not hits:
                return 1
            return max(1, int(window_seconds - (now - hits[0])) + 1)

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()
