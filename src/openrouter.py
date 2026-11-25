"""Shared OpenRouter client.

OpenRouter exposes an OpenAI-compatible API, so the official ``openai`` SDK is used
with a different ``base_url``. One key (OPENROUTER_API_KEY) covers both the
embeddings model and the chat model.
"""

from __future__ import annotations

from openai import OpenAI

from src import config

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        if not config.OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY is not set. Add it to the .env file.")
        _client = OpenAI(
            base_url=config.OPENROUTER_BASE_URL,
            api_key=config.OPENROUTER_API_KEY,
            # Optional attribution headers shown on openrouter.ai
            default_headers={"X-Title": "MiBank RAG demo"},
        )
    return _client
