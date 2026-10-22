"""
Shared evaluation data.

Holds the hand-labelled ground truth (question + the time window in the video
where the answer lives) and builds a Ragas-ready dataset by pulling the
reference text for each window straight from the transcript segments.

Used by:
  - eval.py        (timestamp-overlap retrieval eval; only needs the windows)
  - eval_ragas.py  (LLM-judged eval; needs the `reference` text too)
"""

from __future__ import annotations

import json
from pathlib import Path

from chunker import load_transcript_json
from transcript import VideoTranscript

TRANSCRIPT_PATH = "data/transcripts/8idr1WZ1A7Q.json"
DATASET_OUT_PATH = "data/eval/eval_dataset.json"

ground_truth = [
    {
        "question": "What are the three sellers and their respective ratings and number of reviews?",
        "answer_start_sec": 6.72,
        "answer_end_sec": 19.56,
    },
    {
        "question": "Why might people be suspicious of a 100% positive rating with only a small number of reviews?",
        "answer_start_sec": 28.12,
        "answer_end_sec": 42.146,
    },
    {
        "question": "What is the trade-off between having more data and having a lower absolute percentage rating?",
        "answer_start_sec": 43.92,
        "answer_end_sec": 53.26,
    },
    {
        "question": "What topics will the three videos cover?",
        "answer_start_sec": 71.56,
        "answer_end_sec": 96.98,
    },
    {
        "question": "What does the suggested rule do to a 10 out of 10 rating?",
        "answer_start_sec": 106.88,
        "answer_end_sec": 120.44,
    },
    {
        "question": "How are the three sellers evaluated using Laplace's rule of succession?",
        "answer_start_sec": 121.5,
        "answer_end_sec": 158.52,
    },
    {
        "question": "What is Laplace's rule of succession?",
        "answer_start_sec": 154.64,
        "answer_end_sec": 170.365,
    },
    {
        "question": "How is the seller's underlying success rate modeled?",
        "answer_start_sec": 186.7,
        "answer_end_sec": 204.10,
    },
    {
        "question": "Why doesn't a 100% rating from 10 reviews necessarily mean that the underlying success rate is 100%?",
        "answer_start_sec": 205.64,
        "answer_end_sec": 213.92,
    },
    {
        "question": "How does the simulation demonstrate that a seller with a 95% success rate can receive 10 positive reviews out of 10?",
        "answer_start_sec": 213.92,
        "answer_end_sec": 254.46,
    },
    {
        "question": "What is the goal when choosing between sellers despite uncertainty about their true success rates?",
        "answer_start_sec": 261.6,
        "answer_end_sec": 283.92,
    },
    {
        "question": "How does uncertainty about a long-run frequency differ from ordinary probability examples such as coin flips?",
        "answer_start_sec": 285.12,
        "answer_end_sec": 302.04,
    },
    {
        "question": "How can the same probability model be applied to estimating the error rate of cars produced in a factory?",
        "answer_start_sec": 303.16,
        "answer_end_sec": 336.66,
    },
    {
        "question": "What is the probability of observing the data given an assumed success rate?",
        "answer_start_sec": 340.84,
        "answer_end_sec": 357.12,
    },
    {
        "question": "How does the binomial formula calculate the probability of getting exactly 48 positive reviews out of 50?",
        "answer_start_sec": 409.2,
        "answer_end_sec": 476.792,
    },
    {
        "question": "What is the binomial distribution and when does it occur?",
        "answer_start_sec": 494.76,
        "answer_end_sec": 516.72,
    },
    {
        "question": "What happens to the probability of observing 48 positive reviews out of 50 as the assumed success rate changes?",
        "answer_start_sec": 534.06,
        "answer_end_sec": 609.608,
    },
    {
        "question": "Why can't we simply use the most likely success rate as the probability of having a good experience?",
        "answer_start_sec": 645.48,
        "answer_end_sec": 706.855,
    },
]


def extract_reference(
    transcript: VideoTranscript,
    start_sec: float,
    end_sec: float,
) -> str:
    """
    Return the transcript text spoken in [start_sec, end_sec].

    A segment is included if its time span overlaps the window at all, so
    segments straddling a boundary are kept whole (we never cut mid-segment).
    """
    parts = []
    for seg in transcript.segments:
        seg_start = seg.start
        seg_end = seg.start + seg.duration
        if seg_start <= end_sec and start_sec <= seg_end:
            parts.append(seg.text.strip().replace("\n", " "))
    return " ".join(parts)


def build_eval_dataset(
    transcript: VideoTranscript,
    items: list[dict] | None = None,
) -> list[dict]:
    """
    Attach a `reference` answer (taken from the transcript) to every
    ground-truth item. Returns new dicts; the inputs are not mutated.

    Raises:
        ValueError: if any window yields no transcript text (bad timestamps).
    """
    items = ground_truth if items is None else items
    dataset = []
    for i, item in enumerate(items):
        reference = extract_reference(
            transcript, item["answer_start_sec"], item["answer_end_sec"]
        )
        if not reference:
            raise ValueError(
                f"Item {i} ({item['question']!r}): no transcript text in "
                f"{item['answer_start_sec']}s-{item['answer_end_sec']}s."
            )
        dataset.append({**item, "reference": reference})
    return dataset


def load_eval_dataset(
    transcript_path: str | Path = TRANSCRIPT_PATH,
) -> tuple[VideoTranscript, list[dict]]:
    """Convenience: load the transcript and build the dataset in one call."""
    transcript = load_transcript_json(transcript_path)
    return transcript, build_eval_dataset(transcript)


def save_eval_dataset(dataset: list[dict], path: str | Path = DATASET_OUT_PATH) -> Path:
    """Save the dataset as JSON so it can be inspected/hand-corrected."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataset, indent=2, ensure_ascii=False))
    return path


if __name__ == "__main__":
    transcript, dataset = load_eval_dataset()
    out = save_eval_dataset(dataset)
    print(f"Built {len(dataset)} items -> {out}\n")

    for i, item in enumerate(dataset):
        words = len(item["reference"].split())
        print(f"[{i:02d}] {item['answer_start_sec']:.1f}-{item['answer_end_sec']:.1f}s "
              f"({words} words)")
        print(f"     Q: {item['question']}")
        print(f"     R: {item['reference'][:140]}...\n")