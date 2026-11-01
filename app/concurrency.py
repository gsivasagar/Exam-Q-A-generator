"""Bounded parallelism for independent LLM calls."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional, Sequence, TypeVar

from . import config

T = TypeVar("T")
R = TypeVar("R")
HARD_CAP = 16


def effective_workers(llm=None) -> int:
    """How many calls may run at once: the provider's own limit (`max_concurrency`), else
    LLM_CONCURRENCY, never above HARD_CAP and never below 1."""
    declared = getattr(llm, "max_concurrency", None)
    n = int(declared) if declared else config.LLM_CONCURRENCY
    return max(1, min(n, HARD_CAP))


def run_parallel(fn: Callable[[T], R], items: Sequence[T], workers: int,
                 on_done: Optional[Callable[[int, int], None]] = None) -> List[R]:
    """fn(item) for every item, results in input order. With workers <= 1 (or one item) it runs inline
    in the calling thread. `on_done(done, total)` is always called from the calling thread, so it is
    safe to touch UI objects from it. The first exception is re-raised after the pool has drained."""
    total = len(items)
    results: List[Optional[R]] = [None] * total
    if workers <= 1 or total <= 1:
        for i, item in enumerate(items):
            results[i] = fn(item)
            if on_done:
                on_done(i + 1, total)
        return results  # type: ignore[return-value]
    from concurrent.futures import as_completed

    with ThreadPoolExecutor(max_workers=min(workers, total)) as pool:
        futures = {pool.submit(fn, item): i for i, item in enumerate(items)}
        done = 0
        first_error: Optional[BaseException] = None
        for fut in as_completed(futures):
            try:
                results[futures[fut]] = fut.result()
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                first_error = first_error or exc
            done += 1
            if on_done:
                on_done(done, total)
    if first_error is not None:
        raise first_error
    return results  # type: ignore[return-value]
