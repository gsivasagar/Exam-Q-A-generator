"""Optional response cache for any LLM provider (set LLM_CACHE_DIR to enable).

Useful for development, tests and evaluation runs: an identical request (model, prompt, schema,
temperature) returns the stored reply instead of calling the model. Failures are never cached.
Note: with the cache on, an identical prompt gives an identical quiz."""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Dict, Optional


class CachedProvider:
    def __init__(self, inner, directory: Optional[str] = None):
        self._inner = inner
        self._dir = Path(directory) if directory else None
        if self._dir:
            self._dir.mkdir(parents=True, exist_ok=True)
        self._memory: Dict[str, str] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def __getattr__(self, name):  # model, stats, usage_summary(), context_chars, max_concurrency, ...
        return getattr(self._inner, name)

    def _key(self, prompt: str, schema, temperature) -> str:
        blob = json.dumps([getattr(self._inner, "model", ""), prompt, schema, temperature],
                          sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def generate(self, prompt: str, *, schema: Optional[dict] = None, temperature: Optional[float] = None) -> str:
        key = self._key(prompt, schema, temperature)
        with self._lock:
            cached = self._memory.get(key)
            if cached is None and self._dir and (self._dir / f"{key}.json").exists():
                try:
                    cached = json.loads((self._dir / f"{key}.json").read_text("utf-8"))["response"]
                    self._memory[key] = cached
                except (OSError, ValueError, KeyError):
                    cached = None
            if cached is not None:
                self.hits += 1
                return cached
        reply = self._inner.generate(prompt, schema=schema, temperature=temperature)
        with self._lock:
            self.misses += 1
            self._memory[key] = reply
            if self._dir:
                try:
                    (self._dir / f"{key}.json").write_text(json.dumps({"response": reply}), "utf-8")
                except OSError:
                    pass  # a read-only cache directory must not break generation
        return reply
