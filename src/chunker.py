"""
Milestone 2: Chunking.

Splits a VideoTranscript's segments into overlapping text chunks suitable
for embedding, while preserving start/end timestamps so each chunk can be
traced back to a point in the source video.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from transcript import TranscriptSegment, VideoTranscript


@dataclass(frozen=True)
class TranscriptChunk:
    video_id: str
    chunk_index: int
    text: str
    start: float
    end: float

    def to_dict(self) -> dict:
        return asdict(self)


def load_transcript_json(path: str | Path) -> VideoTranscript:
    """Load a VideoTranscript previously saved by transcript.save_transcript()."""
    data = json.loads(Path(path).read_text())
    segments = [TranscriptSegment(**seg) for seg in data["segments"]]
    return VideoTranscript(
        video_id=data["video_id"],
        url=data["url"],
        language=data["language"],
        language_code=data["language_code"],
        is_generated=data["is_generated"],
        segments=segments,
    )


def chunk_transcript(
    transcript: VideoTranscript,
    chunk_size_chars: int = 1000,
    chunk_overlap_chars: int = 200,
) -> list[TranscriptChunk]:
    """
    Group transcript segments into overlapping text chunks.

    Segments are concatenated in order until the accumulated text reaches
    ~chunk_size_chars, at which point the chunk is closed off. The next
    chunk starts a bit before the previous one ended (~chunk_overlap_chars
    worth of trailing segments), so context isn't lost at chunk boundaries.

    Args:
        transcript: The VideoTranscript to chunk.
        chunk_size_chars: Target chunk size in characters.
        chunk_overlap_chars: Overlap between consecutive chunks, in characters.

    Raises:
        ValueError: if chunk_overlap_chars >= chunk_size_chars.
    """
    if chunk_overlap_chars >= chunk_size_chars:
        raise ValueError("chunk_overlap_chars must be smaller than chunk_size_chars")

    segments = transcript.segments
    if not segments:
        return []

    n = len(segments)
    chunks: list[TranscriptChunk] = []
    start_idx = 0
    chunk_index = 0

    while start_idx < n:
        # Grow the window forward until we hit the target chunk size.
        end_idx = start_idx
        length = 0
        while end_idx < n and length < chunk_size_chars:
            length += len(segments[end_idx].text) + 1
            end_idx += 1

        chunk_segs = segments[start_idx:end_idx]
        text = " ".join(s.text.strip() for s in chunk_segs)
        chunks.append(
            TranscriptChunk(
                video_id=transcript.video_id,
                chunk_index=chunk_index,
                text=text,
                start=chunk_segs[0].start,
                end=chunk_segs[-1].start + chunk_segs[-1].duration,
            )
        )
        chunk_index += 1

        if end_idx >= n:
            break

        # Back up from end_idx by ~chunk_overlap_chars worth of segments
        # to find where the next chunk should start.
        overlap_len = 0
        back_idx = end_idx
        while back_idx > start_idx and overlap_len < chunk_overlap_chars:
            back_idx -= 1
            overlap_len += len(segments[back_idx].text) + 1

        # Guarantee forward progress even if overlap covers the whole window.
        start_idx = back_idx if back_idx > start_idx else end_idx

    return chunks


def save_chunks(chunks: list[TranscriptChunk], output_dir: str | Path, video_id: str) -> Path:
    """Save chunks as pretty-printed JSON. Returns the file path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{video_id}_chunks.json"
    out_path.write_text(
        json.dumps([c.to_dict() for c in chunks], indent=2, ensure_ascii=False)
    )
    return out_path


if __name__ == "__main__":
    import sys

    #if len(sys.argv) != 2:
    #    print("Usage: python chunking.py <path-to-transcript-json>")
    #    sys.exit(1)

    transcript_path = "D:/Project/yt-qna-rag/data/transcripts/8idr1WZ1A7Q.json"
    transcript = load_transcript_json(transcript_path)
    chunks = chunk_transcript(transcript, chunk_size_chars=1000, chunk_overlap_chars=200)
    path = save_chunks(chunks, "data/chunks", transcript.video_id)

    print(f"Created {len(chunks)} chunks -> {path}")
    for c in chunks[:3]:
        preview = c.text[:80].replace("\n", " ")
        print(f"[{c.start:.1f}s - {c.end:.1f}s] {preview}...")