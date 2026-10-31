"""Thin provider abstraction so callers don't branch on the provider name."""
from __future__ import annotations

from typing import Optional, Protocol

from . import config


class LLMError(RuntimeError):
    """Raised when the LLM backend cannot be reached or returns an error."""


class LLMProvider(Protocol):
    def generate(self, prompt: str, *, schema: Optional[dict] = None) -> str: ...


class OllamaProvider:
    def __init__(self, model: Optional[str] = None, host: Optional[str] = None):
        from ollama import Client

        self.model = model or config.OLLAMA_MODEL
        self._client = Client(host=host or config.OLLAMA_URL)

    def generate(self, prompt: str, *, schema: Optional[dict] = None) -> str:
        """If `schema` (a JSON schema dict) is given, output is constrained to it.
        Plain format="json" is avoided: it forces a single object, not a list."""
        try:
            kwargs = {"format": schema} if schema else {}
            resp = self._client.generate(model=self.model, prompt=prompt, stream=False, **kwargs)
            return resp["response"]
        except Exception as exc:  # connection refused, model missing, ...
            raise LLMError(
                f"Ollama request failed ({exc}). Is Ollama running and is "
                f"'{self.model}' pulled?"
            ) from exc


class GeminiProvider:
    def __init__(self, api_key: str, model: Optional[str] = None):
        if not api_key:
            raise LLMError("Gemini API key is required when provider is Gemini.")
        from google import genai

        self.model = model or config.GEMINI_MODEL
        self._client = genai.Client(api_key=api_key)

    def generate(self, prompt: str, *, schema: Optional[dict] = None) -> str:
        try:
            cfg = {"response_mime_type": "application/json"} if schema else None
            resp = self._client.models.generate_content(
                model=self.model, contents=prompt, config=cfg
            )
            return resp.text or ""
        except Exception as exc:
            raise LLMError(f"Gemini request failed: {exc}") from exc


def get_provider(
    provider: str = "Ollama",
    *,
    gemini_api_key: Optional[str] = None,
    ollama_model: Optional[str] = None,
) -> LLMProvider:
    if provider == "Gemini":
        return GeminiProvider(gemini_api_key or config.GEMINI_API_KEY)
    return OllamaProvider(model=ollama_model)
