"""Builds the model provider for a request. A separate module so tests can swap in a fake."""

from __future__ import annotations

from ..llm import GeminiProvider, LLMProvider, get_provider
from .deps import ModelCredentials
from .errors import ApiProblem


def make_llm(creds: ModelCredentials) -> LLMProvider:
    if creds.provider == "Gemini" and not creds.api_key:
        raise ApiProblem("needs_gemini", "Enter your Gemini API key, or choose Ollama.")
    return get_provider(creds.provider, gemini_api_key=creds.api_key)


def make_vision(creds: ModelCredentials):
    """A Gemini provider for reading scans and images, or None when the caller is not using Gemini."""
    if creds.provider == "Gemini" and creds.api_key:
        return GeminiProvider(creds.api_key)
    return None
