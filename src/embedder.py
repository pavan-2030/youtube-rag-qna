"""
v2: Embedding + storage — now backed by Qdrant Cloud.
 
Takes chunks produced by chunker.py, embeds them with Google's embedding
model, and upserts them into a Qdrant collection so they can be
semantically searched later (Milestone 4: retrieval + QA).
 
Migration note (Chroma -> Qdrant):
- Chroma let us use arbitrary string IDs ("{video_id}_{chunk_index}").
  Qdrant point IDs must be unsigned ints or UUIDs, so we derive a
  deterministic UUID5 from that same string instead. Re-embedding a video
  still overwrites its old points rather than duplicating them.
- Chroma's collection.get(where=...) existence check is replaced by
  is_video_indexed(), which uses Qdrant's scroll + filter API.
- Metadata is called "payload" in Qdrant; we also store the chunk text
  in the payload (Chroma stored documents separately from metadata, but
  Qdrant only has one bucket per point).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from chunker import TranscriptChunk

load_dotenv()

DEFAULT_COLLECTION_NAME = "yt_transcripts"
DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-001"
DEFAULT_BATCH_SIZE = 100
DEFAULT_EMBEDDING_DIM = 3072

#Fixed namespace so uuid5(_ID_NAMESPACE, key) is stable across runs/machines.
_ID_NAMESPACE = uuid.UUID("2f4a6b8e-0000-4000-8000-000000000000")


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

def get_qdrant_client() -> QdrantClient:
    """
    Build the Qdrant client from environment variables.
 
    Requires QDRANT_URL and QDRANT_API_KEY (from your Qdrant Cloud cluster
    dashboard) to be set, e.g. via a .env file loaded by python-dotenv.
    """
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if not url or not api_key:
        raise EmbeddingError(
            "QDRANT_URL and QDRANT_API_KEY must be set. Add them to your .env file."
        )
    return QdrantClient(url=url, api_key=api_key)


def get_collection(
    collection_name: str = DEFAULT_COLLECTION_NAME,
    vector_size: int = DEFAULT_EMBEDDING_DIM,
) -> QdrantClient:
    """
    Get (or create) the Qdrant collection, creating it on first use.
 
    Returns the connected QdrantClient rather than a Chroma-style
    Collection object; callers pass collection_name explicitly on every
    subsequent call. This keeps the function name/shape familiar to
    main.py / app.py even though the underlying client changed.
    """
    client = get_qdrant_client()
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
    return client

def _point_id(video_id: str, chunk_index: int) -> str:
    """Deterministic UUID5 so re-embedding a video overwrites, not duplicates."""
    return str(uuid.uuid5(_ID_NAMESPACE, f"{video_id}_{chunk_index}"))
 
 
def is_video_indexed(
    video_id: str,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> bool:
    """
    Check whether a video's chunks are already stored in the collection.
 
    Replaces the old Chroma-specific `collection.get(where=...)` check
    that used to live inline in main.py / app.py.
    """
    client = get_qdrant_client()
    if not client.collection_exists(collection_name):
        return False
    hits, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=Filter(
            must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))]
        ),
        limit=1,
    )
    return len(hits) > 0


def _batched(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def embed_and_store_chunks(
    chunks: list[TranscriptChunk],
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> QdrantClient:
    """
    Embed a list of TranscriptChunks and upsert them into Qdrant.
 
    Point IDs are deterministic UUID5s derived from
    f"{video_id}_{chunk_index}", so re-running this on the same video
    safely overwrites existing entries instead of duplicating them.
 
    Raises:
        EmbeddingError: if chunks is empty or embedding/storage fails.
    """
    if not chunks:
        raise EmbeddingError("No chunks provided to embed.")

    embedder = get_embeddings_client(model=embedding_model)
    client = get_collection(collection_name)

    for batch in _batched(chunks, batch_size):
        documents = [c.text for c in batch]
 
        try:
            vectors = embedder.embed_documents(documents, task_type="RETRIEVAL_DOCUMENT")
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(f"Failed to embed batch: {exc}") from exc
 
        points = [
            PointStruct(
                id=_point_id(c.video_id, c.chunk_index),
                vector=vec,
                payload={
                    "video_id": c.video_id,
                    "chunk_index": c.chunk_index,
                    "start": c.start,
                    "end": c.end,
                    "text": c.text,
                },
            )
            for c, vec in zip(batch, vectors)
        ]
 
        try:
            client.upsert(collection_name=collection_name, points=points)
        except Exception as exc:
            raise EmbeddingError(f"Failed to upsert batch into Qdrant: {exc}") from exc
 
    return client


def embed_and_store_from_file(
    chunks_path: str | Path,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> QdrantClient:
    """Convenience wrapper: chunks JSON path in, populated collection out."""
    chunks = load_chunks_json(chunks_path)
    return embed_and_store_chunks(
        chunks,
        collection_name=collection_name,
        embedding_model=embedding_model,
    )


if __name__ == "__main__":
    chunks_file = "data/chunks/8idr1WZ1A7Q_chunks.json"
    try:
        client = embed_and_store_from_file(chunks_file)
    except EmbeddingError as e:
        print(f"Error: {e}")
        raise SystemExit(1)
 
    # Quick sanity-check query using the same embedder, to confirm retrieval works.
    embedder = get_embeddings_client()
    query_vec = embedder.embed_query(
        "why does more data increase confidence in an estimate?",
        task_type="RETRIEVAL_QUERY",
    )
    results = client.query_points(
        collection_name=DEFAULT_COLLECTION_NAME,
        query=query_vec,
        limit=3,
    ).points
    for point in results:
        p = point.payload
        preview = p["text"][:80].replace("\n", " ")
        print(f"[{p['start']:.1f}s - {p['end']:.1f}s] (score={point.score:.3f}) {preview}...")