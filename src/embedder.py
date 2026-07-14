"""
Milestone 3: Embedding + storage.

Takes chunks produced by chunker.py, embeds them with Google's embedding
model, and upserts them into a persistent ChromaDB collection so they can
be semantically searched later (Milestone 4: retrieval + QA).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from chunker import TranscriptChunk

load_dotenv()

DEFAULT_PERSIST_DIR = "data/chroma"
DEFAULT_COLLECTION_NAME = "yt_transcripts"
DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-001"
DEFAULT_BATCH_SIZE = 100


class EmbeddingError(Exception):
    """Raised when chunks could not be embedded or stored."""


def load_chunks_json(path: str | Path) -> list[TranscriptChunk]:
    """Load chunks previously saved by chunker.save_chunks()."""
    data = json.loads(Path(path).read_text())
    return [TranscriptChunk(**c) for c in data]


def get_embeddings_client(
    model: str = DEFAULT_EMBEDDING_MODEL,
) -> GoogleGenerativeAIEmbeddings:
    """
    Build the embeddings client.

    Requires GOOGLE_API_KEY to be set (e.g. via a .env file loaded by
    python-dotenv).
    """
    if not os.getenv("GOOGLE_API_KEY"):
        raise EmbeddingError(
            "GOOGLE_API_KEY is not set. Add it to your .env file or environment."
        )
    return GoogleGenerativeAIEmbeddings(model=model)


def get_collection(
    persist_directory: str | Path = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> Collection:
    """Get (or create) a persistent Chroma collection."""
    Path(persist_directory).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_directory))
    # We supply our own embeddings at write time, so no embedding_function
    # is registered on the collection itself.
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def _batched(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def embed_and_store_chunks(
    chunks: list[TranscriptChunk],
    persist_directory: str | Path = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Collection:
    """
    Embed a list of TranscriptChunks and upsert them into ChromaDB.

    IDs are deterministic (f"{video_id}_{chunk_index}"), so re-running this
    on the same video safely overwrites existing entries instead of
    duplicating them.

    Raises:
        EmbeddingError: if chunks is empty or embedding/storage fails.
    """
    if not chunks:
        raise EmbeddingError("No chunks provided to embed.")

    embedder = get_embeddings_client(model=embedding_model)
    collection = get_collection(persist_directory, collection_name)

    for batch in _batched(chunks, batch_size):
        ids = [f"{c.video_id}_{c.chunk_index}" for c in batch]
        documents = [c.text for c in batch]
        metadatas = [
            {
                "video_id": c.video_id,
                "chunk_index": c.chunk_index,
                "start": c.start,
                "end": c.end,
            }
            for c in batch
        ]

        try:
            vectors = embedder.embed_documents(documents, task_type="RETRIEVAL_DOCUMENT")
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(f"Failed to embed batch: {exc}") from exc

        try:
            collection.upsert(
                ids=ids,
                embeddings=vectors,
                documents=documents,
                metadatas=metadatas,
            )
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(f"Failed to upsert batch into Chroma: {exc}") from exc

    return collection


def embed_and_store_from_file(
    chunks_path: str | Path,
    persist_directory: str | Path = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> Collection:
    """Convenience wrapper: chunks JSON path in, populated Collection out."""
    chunks = load_chunks_json(chunks_path)
    return embed_and_store_chunks(
        chunks,
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_model=embedding_model,
    )


if __name__ == "__main__":
    chunks_file = "data/chunks/8idr1WZ1A7Q_chunks.json"
    try:
        collection = embed_and_store_from_file(chunks_file)
    except EmbeddingError as e:
        print(f"Error: {e}")
        raise SystemExit(1)

    #print(f"Collection '{collection.name}' now has {collection.count()} vectors.")

    # Quick sanity-check query using the same embedder, to confirm retrieval works.
    embedder = get_embeddings_client()
    query_vec = embedder.embed_query(
        "why does more data increase confidence in an estimate?",
        task_type="RETRIEVAL_QUERY",
    )
    results = collection.query(query_embeddings=[query_vec], n_results=3)
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        preview = doc[:80].replace("\n", " ")
        print(f"[{meta['start']:.1f}s - {meta['end']:.1f}s] (dist={dist:.3f}) {preview}...")