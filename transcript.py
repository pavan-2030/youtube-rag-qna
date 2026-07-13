"""
Milestone 1: Transcript extraction.

Given a YouTube video URL, extract its transcript (with timestamps) and
save it as a JSON file for downstream chunking / embedding.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    CouldNotRetrieveTranscript,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)


class TranscriptExtractionError(Exception):
    """Raised when a transcript cannot be extracted for a video."""


@dataclass(frozen=True)
class TranscriptSegment:
    text: str
    start: float
    duration: float


@dataclass(frozen=True)
class VideoTranscript:
    video_id: str
    url: str
    language: str
    language_code: str
    is_generated: bool
    segments: list[TranscriptSegment]

    def full_text(self) -> str:
        """Concatenate all segments into a single plain-text blob."""
        return " ".join(seg.text.strip() for seg in self.segments)

    def to_dict(self) -> dict:
        return {
            "video_id": self.video_id,
            "url": self.url,
            "language": self.language,
            "language_code": self.language_code,
            "is_generated": self.is_generated,
            "segments": [asdict(seg) for seg in self.segments],
        }


# Matches the video ID out of the various URL shapes YouTube uses:
#   https://www.youtube.com/watch?v=VIDEO_ID
#   https://youtu.be/VIDEO_ID
#   https://www.youtube.com/embed/VIDEO_ID
#   https://www.youtube.com/shorts/VIDEO_ID
_VIDEO_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{11}$")


def extract_video_id(url: str) -> str:
    """
    Extract the 11-character YouTube video ID from a full URL.

    Raises:
        ValueError: if no valid video ID could be found in the URL.
    """
    url = url.strip()

    # If someone already passed a bare video ID, accept it.
    if _VIDEO_ID_RE.match(url):
        return url

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower().removeprefix("www.")

    video_id: str | None = None

    if host in {"youtu.be"}:
        # https://youtu.be/VIDEO_ID
        video_id = parsed.path.lstrip("/").split("/")[0]
    elif host in {"youtube.com", "m.youtube.com", "music.youtube.com"}:
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [None])[0]
        elif parsed.path.startswith(("/embed/", "/shorts/", "/live/")):
            video_id = parsed.path.split("/")[2] if len(parsed.path.split("/")) > 2 else None

    if not video_id or not _VIDEO_ID_RE.match(video_id):
        raise ValueError(f"Could not extract a valid video ID from URL: {url!r}")

    return video_id


def fetch_transcript(
    video_id: str,
    languages: tuple[str, ...] = ("en",),
) -> VideoTranscript:
    """
    Fetch the transcript for a given video ID.

    Args:
        video_id: The 11-character YouTube video ID.
        languages: Preferred languages, in priority order. Falls back to
            whatever is available if none of these match.

    Raises:
        TranscriptExtractionError: if no transcript could be retrieved.
    """
    api = YouTubeTranscriptApi()
    try:
        fetched = api.fetch(video_id, languages=list(languages))
    except TranscriptsDisabled as exc:
        raise TranscriptExtractionError(
            f"Transcripts are disabled for video {video_id!r}."
        ) from exc
    except NoTranscriptFound as exc:
        raise TranscriptExtractionError(
            f"No transcript found for video {video_id!r} in languages {languages!r}."
        ) from exc
    except VideoUnavailable as exc:
        raise TranscriptExtractionError(
            f"Video {video_id!r} is unavailable."
        ) from exc
    except CouldNotRetrieveTranscript as exc:
        raise TranscriptExtractionError(
            f"Could not retrieve transcript for video {video_id!r}: {exc}"
        ) from exc

    segments = [
        TranscriptSegment(text=s.text, start=s.start, duration=s.duration)
        for s in fetched.snippets
    ]

    return VideoTranscript(
        video_id=video_id,
        url=f"https://www.youtube.com/watch?v={video_id}",
        language=fetched.language,
        language_code=fetched.language_code,
        is_generated=fetched.is_generated,
        segments=segments,
    )


def extract_transcript_from_url(
    url: str,
    languages: tuple[str, ...] = ("en",),
) -> VideoTranscript:
    """Convenience wrapper: URL in, VideoTranscript out."""
    video_id = extract_video_id(url)
    return fetch_transcript(video_id, languages=languages)


def save_transcript(transcript: VideoTranscript, output_dir: str | Path) -> Path:
    """Save a VideoTranscript as pretty-printed JSON. Returns the file path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{transcript.video_id}.json"
    out_path.write_text(json.dumps(transcript.to_dict(), indent=2, ensure_ascii=False))
    return out_path


if __name__ == "__main__":
    import sys

    #if len(sys.argv) != 2:
    #    print("Usage: python -m yt_qna_rag.transcript <youtube_url>")
    #    sys.exit(1)

    video_url = "https://youtu.be/8idr1WZ1A7Q?si=IryKjyEGILASj25M"
    try:
        result = extract_transcript_from_url(video_url)
    except (ValueError, TranscriptExtractionError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    path = save_transcript(result, "data/transcripts")
    print(f"Saved transcript for video {result.video_id} -> {path}")
    print(f"Language: {result.language} ({result.language_code}), generated={result.is_generated}")
    print(f"Segments: {len(result.segments)}")