"""ChromaDB persistent vector store (embeddings are computed by us, not by Chroma)."""

from __future__ import annotations

import chromadb

from src import config


def _client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def get_collection() -> chromadb.Collection:
    return _client().get_or_create_collection(
        name=config.COLLECTION_NAME,
        configuration={"hnsw": {"space": "cosine"}},
        embedding_function=None,
    )


def reset_collection() -> chromadb.Collection:
    client = _client()
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:  # collection does not exist yet
        pass
    return get_collection()


def count() -> int:
    return get_collection().count()
