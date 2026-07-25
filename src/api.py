"""
Milestone 6: FastAPI web UI layer.

Wraps the existing pipeline (transcript -> chunk -> embed -> retrieve/answer)
in an HTTP API so a frontend (or curl / Swagger UI) can drive it without a
local terminal session.
"""

from __future__ import annotations

from enum import Enum

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from chunker import chunk_transcript
from embedder import (
    DEFAULT_COLLECTION_NAME,
    EmbeddingError,
    embed_and_store_chunks,
    is_video_indexed,
)
from retriever import RetrievalError, ask_question
from transcript import (
    TranscriptExtractionError,
    extract_transcript_from_url,
    extract_video_id,
)

app = FastAPI(title="YouTube Q&A RAG API", version="0.1.0")

# Loosen this to your actual frontend origin before deploying.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# In-memory job status store.
#
# Good enough for a single-instance demo deployment. If this ever runs
# behind multiple workers/replicas, swap this dict for Redis (or Qdrant
# itself, via a "status" payload field) since state needs to be shared
# across processes.
# ---------------------------------------------------------------------------
class VideoStatus(str, Enum):
    INDEXING = "indexing"
    READY = "ready"
    ERROR = "error"


_JOBS: dict[str, dict] = {}  # video_id -> {"status": VideoStatus, "detail": str | None}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class VideoRequest(BaseModel):
    url: str = Field(..., examples=["https://youtu.be/8idr1WZ1A7Q"])


class VideoStatusResponse(BaseModel):
    video_id: str
    status: VideoStatus
    detail: str | None = None


class AskRequest(BaseModel):
    question: str
    top_k: int = 5


class SourceChunk(BaseModel):
    text: str
    start: float
    end: float
    timestamp: str  # "MM:SS-MM:SS", pre-formatted for the frontend


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceChunk]


# ---------------------------------------------------------------------------
# Background indexing task
# ---------------------------------------------------------------------------
def _run_indexing(url: str, video_id: str) -> None:
    try:
        transcript = extract_transcript_from_url(url)
        chunks = chunk_transcript(transcript)
        embed_and_store_chunks(chunks, collection_name=DEFAULT_COLLECTION_NAME)
        _JOBS[video_id] = {"status": VideoStatus.READY, "detail": None}
    except (TranscriptExtractionError, EmbeddingError) as e:
        _JOBS[video_id] = {"status": VideoStatus.ERROR, "detail": str(e)}
    except Exception as e:  # noqa: BLE001
        _JOBS[video_id] = {"status": VideoStatus.ERROR, "detail": f"Unexpected error: {e}"}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.post("/videos", response_model=VideoStatusResponse)
def submit_video(req: VideoRequest, background_tasks: BackgroundTasks) -> VideoStatusResponse:
    """
    Kick off indexing for a video. Returns immediately; poll
    GET /videos/{video_id}/status for progress.
    """
    try:
        video_id = extract_video_id(req.url)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    if is_video_indexed(video_id, DEFAULT_COLLECTION_NAME):
        _JOBS[video_id] = {"status": VideoStatus.READY, "detail": None}
        return VideoStatusResponse(video_id=video_id, status=VideoStatus.READY)

    # Already indexing (e.g. duplicate submit) — don't kick off a second job.
    existing = _JOBS.get(video_id)
    if existing and existing["status"] == VideoStatus.INDEXING:
        return VideoStatusResponse(video_id=video_id, status=VideoStatus.INDEXING)

    _JOBS[video_id] = {"status": VideoStatus.INDEXING, "detail": None}
    background_tasks.add_task(_run_indexing, req.url, video_id)
    return VideoStatusResponse(video_id=video_id, status=VideoStatus.INDEXING)


@app.get("/videos/{video_id}/status", response_model=VideoStatusResponse)
def get_video_status(video_id: str) -> VideoStatusResponse:
    """Poll this while a video is indexing."""
    job = _JOBS.get(video_id)

    if job is None:
        # Not tracked in this process's memory (e.g. server restarted,
        # or it was indexed in a previous session) — fall back to Qdrant.
        if is_video_indexed(video_id, DEFAULT_COLLECTION_NAME):
            return VideoStatusResponse(video_id=video_id, status=VideoStatus.READY)
        raise HTTPException(status_code=404, detail="Unknown video_id.")

    return VideoStatusResponse(video_id=video_id, status=job["status"], detail=job["detail"])


@app.post("/videos/{video_id}/ask", response_model=AskResponse)
def ask(video_id: str, req: AskRequest) -> AskResponse:
    """Ask a question about an already-indexed video."""
    job = _JOBS.get(video_id)
    ready = (job and job["status"] == VideoStatus.READY) or is_video_indexed(
        video_id, DEFAULT_COLLECTION_NAME
    )
    if not ready:
        raise HTTPException(
            status_code=409,
            detail="Video is not indexed yet. Check /videos/{video_id}/status.",
        )

    try:
        result = ask_question(req.question, top_k=req.top_k, video_id=video_id)
    except RetrievalError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    def _ts(seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        return f"{m:02d}:{s:02d}"

    sources = [
        SourceChunk(
            text=c.text,
            start=c.start,
            end=c.end,
            timestamp=f"{_ts(c.start)}-{_ts(c.end)}",
        )
        for c in result.sources
    ]
    return AskResponse(question=result.question, answer=result.answer, sources=sources)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}