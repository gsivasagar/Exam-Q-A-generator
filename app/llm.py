"""Thin provider abstraction so callers don't branch on the provider name.

Every provider call goes through `_call`, which adds retries with backoff for transient
errors and records token usage and latency (see `usage_summary`).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Protocol, TypeVar

from . import config

log = logging.getLogger(__name__)

T = TypeVar("T")

_TRANSIENT_STATUS = {408, 425, 429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    """Raised when the LLM backend cannot be reached or returns an error."""


class LLMProvider(Protocol):
    def generate(self, prompt: str, *, schema: Optional[dict] = None,
                 temperature: Optional[float] = None) -> str: ...


@dataclass
class CallStats:
    """Token usage and wall-clock time of one successful LLM call."""
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    seconds: float


def is_transient(exc: BaseException) -> bool:
    """True for errors worth retrying: network/timeouts and 408/429/5xx responses."""
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    try:
        import httpx
        if isinstance(exc, (httpx.TransportError, httpx.TimeoutException)):
            return True
    except ImportError:  # pragma: no cover - httpx ships with ollama/google-genai
        pass
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(exc, "code", None)
    return isinstance(status, int) and status in _TRANSIENT_STATUS


def with_retries(fn: Callable[[], T], *, retries: int, backoff: float,
                 sleep: Callable[[float], None] = time.sleep) -> T:
    """Call fn(); on a transient error retry up to `retries` times, doubling the wait each time."""
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:
            if attempt >= retries or not is_transient(exc):
                raise
            wait = backoff * (2 ** attempt)
            attempt += 1
            log.warning("Transient LLM error (%s); retry %d/%d in %.1fs", exc, attempt, retries, wait)
            sleep(wait)


class _BaseProvider:
    """Shared retry + usage bookkeeping."""

    def __init__(self, retries: Optional[int] = None, backoff: Optional[float] = None):
        self.retries = config.LLM_MAX_RETRIES if retries is None else retries
        self.backoff = config.LLM_RETRY_BACKOFF if backoff is None else backoff
        self.stats: List[CallStats] = []

    def _call(self, fn: Callable[[], T]) -> T:
        return with_retries(fn, retries=self.retries, backoff=self.backoff)

    def _record(self, started: float, prompt_tokens, completion_tokens) -> None:
        stat = CallStats(prompt_tokens, completion_tokens, time.perf_counter() - started)
        self.stats.append(stat)
        log.info("LLM call: %s prompt tokens, %s completion tokens, %.2fs",
                 prompt_tokens, completion_tokens, stat.seconds)

    def usage_summary(self) -> dict:
        """Totals over every call made by this provider instance."""
        return {
            "calls": len(self.stats),
            "prompt_tokens": sum(s.prompt_tokens or 0 for s in self.stats),
            "completion_tokens": sum(s.completion_tokens or 0 for s in self.stats),
            "seconds": round(sum(s.seconds for s in self.stats), 3),
        }


class OllamaProvider(_BaseProvider):
    def __init__(self, model: Optional[str] = None, host: Optional[str] = None, *,
                 timeout: Optional[float] = None, num_ctx: Optional[int] = None,
                 retries: Optional[int] = None, backoff: Optional[float] = None):
        super().__init__(retries, backoff)
        from ollama import Client

        self.model = model or config.OLLAMA_MODEL
        self.num_ctx = num_ctx if num_ctx is not None else config.OLLAMA_NUM_CTX
        # ~3 chars/token, leaving a third of the window for instructions and the answer
        self.context_chars = int(self.num_ctx * 2) if self.num_ctx else None
        self._client = Client(host=host or config.OLLAMA_URL,
                              timeout=timeout if timeout is not None else config.LLM_TIMEOUT)

    def generate(self, prompt: str, *, schema: Optional[dict] = None,
                 temperature: Optional[float] = None) -> str:
        """If `schema` (a JSON schema dict) is given, output is constrained to it.
        Plain format="json" is avoided: it forces a single object, not a list."""
        kwargs: dict = {"format": schema} if schema else {}
        options = {}
        if temperature is not None:
            options["temperature"] = temperature
        if self.num_ctx:
            options["num_ctx"] = self.num_ctx
        if options:
            kwargs["options"] = options
        started = time.perf_counter()
        try:
            resp = self._call(lambda: self._client.generate(
                model=self.model, prompt=prompt, stream=False, **kwargs))
            self._record(started, _get(resp, "prompt_eval_count"), _get(resp, "eval_count"))
            return resp["response"]
        except Exception as exc:  # connection refused, model missing, ...
            raise LLMError(
                f"Ollama request failed ({exc}). Is Ollama running and is "
                f"'{self.model}' pulled?"
            ) from exc


class GeminiProvider(_BaseProvider):
    def __init__(self, api_key: str, model: Optional[str] = None, *,
                 thinking_budget: Optional[int] = None,
                 retries: Optional[int] = None, backoff: Optional[float] = None):
        super().__init__(retries, backoff)
        if not api_key:
            raise LLMError("Gemini API key is required when provider is Gemini.")
        from google import genai

        self.model = model or config.GEMINI_MODEL
        self.context_chars = config.GEMINI_CONTEXT_CHARS
        self.thinking_budget = (thinking_budget if thinking_budget is not None
                                else config.GEMINI_THINKING_BUDGET)
        self._client = genai.Client(api_key=api_key)

    def _config(self, schema: Optional[dict], temperature: Optional[float]):
        from google.genai import types

        kwargs: dict = {}
        if schema:
            kwargs["response_mime_type"] = "application/json"
            kwargs["response_json_schema"] = schema
        if temperature is not None:
            kwargs["temperature"] = temperature
        if self.thinking_budget is not None:
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=self.thinking_budget)
        return types.GenerateContentConfig(**kwargs) if kwargs else None

    def generate(self, prompt: str, *, schema: Optional[dict] = None,
                 temperature: Optional[float] = None) -> str:
        started = time.perf_counter()
        try:
            cfg = self._config(schema, temperature)
            resp = self._call(lambda: self._client.models.generate_content(
                model=self.model, contents=prompt, config=cfg))
            usage = getattr(resp, "usage_metadata", None)
            self._record(started, getattr(usage, "prompt_token_count", None),
                         getattr(usage, "candidates_token_count", None))
            return resp.text or ""
        except Exception as exc:
            raise LLMError(f"Gemini request failed: {exc}") from exc


def _get(resp, key):
    """Read a field from an ollama response (pydantic object or dict)."""
    try:
        return resp[key]
    except (KeyError, TypeError, AttributeError):
        return getattr(resp, key, None)


def get_provider(
    provider: str = "Ollama",
    *,
    gemini_api_key: Optional[str] = None,
    ollama_model: Optional[str] = None,
) -> LLMProvider:
    if provider == "Gemini":
        return GeminiProvider(gemini_api_key or config.GEMINI_API_KEY)
    return OllamaProvider(model=ollama_model)
