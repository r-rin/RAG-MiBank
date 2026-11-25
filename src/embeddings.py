"""Embeddings via OpenRouter (openai/text-embedding-3-small by default)."""

from __future__ import annotations

from src import config
from src.openrouter import get_client

_BATCH_SIZE = 100


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts, batching requests to the API."""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), _BATCH_SIZE):
        batch = texts[start : start + _BATCH_SIZE]
        response = get_client().embeddings.create(model=config.EMBEDDING_MODEL, input=batch)
        vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
    return vectors


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
