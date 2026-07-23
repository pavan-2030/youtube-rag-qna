"""
Milestone 5: End-to-end CLI.

Paste a YouTube link -> transcript is fetched, chunked, and embedded
(skipping work if it's already been processed) -> then ask questions
about that video in a loop, with source timestamps shown for each answer.
"""

from __future__ import annotations

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
)



def process_video(
    url: str,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> str:
    """
    Run transcript -> chunk -> embed for a URL, unless it's already in the
    collection. Returns the video_id on success.
    """
    print("Fetching transcript...")
    transcript = extract_transcript_from_url(url)

    if is_video_indexed(transcript.video_id, collection_name):
        print(f"Video {transcript.video_id} already indexed, skipping embedding.")
        return transcript.video_id

    print("Chunking transcript...")
    chunks = chunk_transcript(transcript)

    print(f"Embedding {len(chunks)} chunks (this calls the Gemini API, may take a bit)...")
    embed_and_store_chunks(chunks, collection_name=collection_name)

    print("Done indexing.")
    return transcript.video_id


def qa_loop(video_id: str) -> str | None:
    """
    Ask questions about a single video until the user types 'new' or 'quit'.

    Returns:
        None if the user typed 'quit', or a raw string (next URL to process)
        if the user typed 'new'.
    """
    print(f"\nAsk anything about this video (video_id={video_id}).")
    print("Commands: 'new' to load another video, 'quit' to exit.\n")

    while True:
        question = input("Q: ").strip()
        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            return None
        if question.lower() == "new":
            return input("Paste new YouTube URL: ").strip()

        try:
            result = ask_question(question, video_id=video_id)
        except RetrievalError as e:
            print(f"Error: {e}\n")
            continue

        print(f"\nA: {result.answer}\n")
        if result.sources:
            print("Sources:")
            for c in result.sources:
                m1, s1 = divmod(int(c.start), 60)
                m2, s2 = divmod(int(c.end), 60)
                preview = c.text[:80].replace("\n", " ")
                print(f"  [{m1:02d}:{s1:02d}-{m2:02d}:{s2:02d}] {preview}...")
        print()


def main() -> None:
    url = input("Paste a YouTube URL: ").strip()

    while url:
        try:
            video_id = process_video(url)
        except (ValueError, TranscriptExtractionError, EmbeddingError) as e:
            print(f"Error: {e}\n")
            url = input("Paste a YouTube URL: ").strip()
            continue

        try:
            url = qa_loop(video_id)
        except KeyboardInterrupt:
            print("\nExiting.")
            return

    print("Goodbye.")


if __name__ == "__main__":
    main()