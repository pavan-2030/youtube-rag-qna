"""
Milestone 4: Retrieval + Question Answering.

Given a natural-language question, retrieves the most relevant transcript
chunks from ChromaDB and asks a Gemini chat model to answer using only
that context, citing the timestamp(s) it drew from.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_google_genai import ChatGoogleGenerativeAI

from embedder import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_PERSIST_DIR,
    EmbeddingError,
    get_collection,
    get_embeddings_client,
)

DEFAULT_CHAT_MODEL = "gemini-2.5-flash"
DEFAULT_TOP_K = 5

ANSWER_SYSTEM_PROMPT = """\
You are answering questions about a YouTube video using only the transcript \
excerpts provided below. Each excerpt is labeled with its start and end time \
in the video.

Rules:
- Answer using ONLY the information in the excerpts. Do not use outside knowledge.
- If the excerpts don't contain enough information to answer, say so plainly \
instead of guessing.
- When you use information from an excerpt, cite its timestamp in the form \
[MM:SS-MM:SS].
- Be concise and direct.
"""


class RetrievalError(Exception):
    """Raised when retrieval or answer generation fails."""


@dataclass(frozen=True)
class RetrievedChunk:
    video_id: str
    text: str
    start: float
    end: float
    distance: float


@dataclass(frozen=True)
class QAResult:
    question: str
    answer: str
    sources: list[RetrievedChunk]


def _format_timestamp(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes:02d}:{secs:02d}"


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    video_id: str | None = None,
    persist_directory: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model: str | None = None,
) -> list[RetrievedChunk]:
    """
    Embed a query and fetch the top_k most relevant transcript chunks.

    Args:
        query: The natural-language question.
        top_k: How many chunks to retrieve.
        video_id: If set, restrict results to a single video.
        persist_directory / collection_name: Where the Chroma collection lives.
        embedding_model: Override the embedding model (defaults to embedder's default).

    Raises:
        RetrievalError: if embedding or querying Chroma fails.
    """
    kwargs = {"model": embedding_model} if embedding_model else {}
    try:
        embedder = get_embeddings_client(**kwargs)
        query_vec = embedder.embed_query(query, task_type="RETRIEVAL_QUERY")
    except EmbeddingError as exc:
        raise RetrievalError(f"Failed to embed query: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise RetrievalError(f"Failed to embed query: {exc}") from exc

    collection = get_collection(persist_directory, collection_name)

    where = {"video_id": video_id} if video_id else None
    try:
        results = collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            where=where,
        )
    except Exception as exc:  # noqa: BLE001
        raise RetrievalError(f"Failed to query Chroma: {exc}") from exc

    if not results["documents"] or not results["documents"][0]:
        return []

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        chunks.append(
            RetrievedChunk(
                video_id=meta["video_id"],
                text=doc,
                start=meta["start"],
                end=meta["end"],
                distance=dist,
            )
        )
    return chunks


def _build_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for c in chunks:
        ts = f"[{_format_timestamp(c.start)}-{_format_timestamp(c.end)}]"
        blocks.append(f"{ts} {c.text}")
    return "\n\n".join(blocks)


def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
    chat_model: str = DEFAULT_CHAT_MODEL,
) -> str:
    """Ask a Gemini chat model to answer the question using only the given chunks."""
    if not chunks:
        return "I couldn't find anything relevant to that question in this video."

    context = _build_context(chunks)
    llm = ChatGoogleGenerativeAI(model=chat_model)

    messages = [
        ("system", ANSWER_SYSTEM_PROMPT),
        ("human", f"Transcript excerpts:\n\n{context}\n\nQuestion: {question}"),
    ]

    try:
        response = llm.invoke(messages)
    except Exception as exc:  # noqa: BLE001
        raise RetrievalError(f"Failed to generate answer: {exc}") from exc

    return response.content


def ask_question(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    video_id: str | None = None,
    persist_directory: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    chat_model: str = DEFAULT_CHAT_MODEL,
) -> QAResult:
    """
    End-to-end: retrieve relevant chunks, then generate a grounded answer.

    Raises:
        RetrievalError: if retrieval or generation fails.
    """
    chunks = retrieve(
        question,
        top_k=top_k,
        video_id=video_id,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    answer = generate_answer(question, chunks, chat_model=chat_model)
    return QAResult(question=question, answer=answer, sources=chunks)


if __name__ == "__main__":
    question = "Why does more data increase confidence in an estimate?"
    try:
        result = ask_question(question)
    except RetrievalError as e:
        print(f"Error: {e}")
        raise SystemExit(1)

    print(f"Q: {result.question}\n")
    print(f"A: {result.answer}\n")
    print("Sources:")
    for c in result.sources:
        ts = f"{_format_timestamp(c.start)}-{_format_timestamp(c.end)}"
        preview = c.text[:80].replace("\n", " ")
        print(f"  [{ts}] (dist={c.distance:.3f}) {preview}...")