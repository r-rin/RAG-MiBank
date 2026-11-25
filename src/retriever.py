"""Query pipeline, step 1: find the chunks most similar to the question."""

from __future__ import annotations

from dataclasses import dataclass

from src.embeddings import embed_query
from src.vector_store import get_collection


@dataclass
class RetrievedChunk:
    text: str
    source: str
    heading: str
    score: float  # cosine similarity, 1.0 = identical direction


def retrieve(question: str, top_k: int = 4) -> list[RetrievedChunk]:
    collection = get_collection()
    if collection.count() == 0:
        return []

    result = collection.query(
        query_embeddings=[embed_query(question)],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    return [
        RetrievedChunk(
            text=doc,
            source=meta["source"],
            heading=meta["heading"],
            score=1.0 - dist,  # Chroma returns cosine *distance*
        )
        for doc, meta, dist in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]
