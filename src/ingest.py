"""Indexing pipeline: load Markdown -> split -> embed -> store in ChromaDB.

Run:  python -m src.ingest
"""

from __future__ import annotations

from src import config
from src.embeddings import embed_texts
from src.splitter import Chunk, split_markdown
from src.vector_store import reset_collection


def load_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(config.DATA_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        chunks.extend(
            split_markdown(
                text,
                source=path.name,
                max_chars=config.CHUNK_MAX_CHARS,
                overlap_chars=config.CHUNK_OVERLAP_CHARS,
            )
        )
    return chunks


def ingest() -> int:
    chunks = load_chunks()
    if not chunks:
        raise RuntimeError(f"No Markdown documents found in {config.DATA_DIR}")

    vectors = embed_texts([c.text for c in chunks])
    collection = reset_collection()
    collection.add(
        ids=[c.id for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=vectors,
        metadatas=[{"source": c.source, "heading": c.heading, "index": c.index} for c in chunks],
    )
    return len(chunks)


if __name__ == "__main__":
    n = ingest()
    print(f"Indexed {n} chunks from {config.DATA_DIR} into '{config.COLLECTION_NAME}'.")
