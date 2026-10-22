"""
LLM-judged evaluation with Ragas.

Pipeline:
  1. Load ground truth + transcript-derived references (eval_data.py).
  2. Run the real RAG pipeline (ask_question) for every question and cache
     the answer + retrieved contexts.
  3. Score with Ragas, one metric at a time in small batches, saving every
     score as it arrives. Re-running only scores what is missing or NaN, so
     timeouts / crashes / rate limits never force a full redo.
  4. Add timestamp-based metrics (the manual eval's rule) as extra columns so
     we can compare Ragas against them.

Usage:
    python eval_ragas.py              # all questions
    python eval_ragas.py --limit 5    # first 5 questions only (cheap debugging)

NOTE: ts_max_precision assumes the collection was indexed with the default
chunker settings (1000/200). Change CHUNK_KWARGS if you evaluate another config.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)
from ragas.run_config import RunConfig

from chunker import chunk_transcript
from embedder import DEFAULT_COLLECTION_NAME, get_embeddings_client, is_video_indexed
from eval_data import load_eval_dataset
from retriever import ask_question, content_to_text

# Quotas are per model, so generating with one Gemma and judging with the
# other splits the daily budget (and avoids the judge grading its own writing).
GEN_MODEL = "gemma-4-26b-a4b-it"
JUDGE_MODEL = "gemma-4-31b-it"

COLLECTION_NAME = DEFAULT_COLLECTION_NAME
TOP_K = 5
CHUNK_KWARGS: dict = {}  # defaults = 1000/200, how the main collection is indexed

GEN_DELAY_SEC = 2.4      # 30 RPM cap -> 2 s/request, small margin
SCORE_BATCH_SIZE = 6     # samples per evaluate() call; scores saved after each
OUT_DIR = Path("data/eval")

_SCORES_PATH = OUT_DIR / f"ragas_scores_{COLLECTION_NAME}_k{TOP_K}.json"
_REPORT_CSV = OUT_DIR / f"ragas_scores_{COLLECTION_NAME}_k{TOP_K}.csv"
_OLD_METRIC_COLS = (
    "context_recall",
    "llm_context_precision_with_reference",
    "faithfulness",
    "answer_relevancy",
)


# ---------------------------------------------------------------------------
# Small JSON-cache helpers
# ---------------------------------------------------------------------------
def _load_cache(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _save_cache(cache: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=False))


def _is_missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


# ---------------------------------------------------------------------------
# Step 1: run the pipeline and cache its outputs
# ---------------------------------------------------------------------------
def generate_records(
    dataset: list[dict],
    video_id: str,
    collection_name: str = COLLECTION_NAME,
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Run ask_question() for each item. Returns one record per item:
      {"sample": {user_input, retrieved_contexts, response, reference},
       "source_spans": [[start, end], ...]}
    Cached per (collection, top_k) and resumed on re-run.
    """
    cache_path = OUT_DIR / f"ragas_samples_{collection_name}_k{top_k}.json"
    cache = _load_cache(cache_path)

    for rec in cache.values():  # older caches may hold list-shaped answers
        s = rec["sample"]
        s["response"] = content_to_text(s["response"])

    for i, item in enumerate(dataset):
        q = item["question"]
        if q in cache:
            continue

        print(f"[{i + 1}/{len(dataset)}] generating: {q[:70]}")
        result = ask_question(
            q,
            top_k=top_k,
            video_id=video_id,
            collection_name=collection_name,
            chat_model=GEN_MODEL,
        )
        cache[q] = {
            "sample": {
                "user_input": q,
                "retrieved_contexts": [c.text for c in result.sources],
                "response": result.answer,
                "reference": item["reference"],
            },
            "source_spans": [[c.start, c.end] for c in result.sources],
        }
        _save_cache(cache, cache_path)
        time.sleep(GEN_DELAY_SEC)

    return [cache[item["question"]] for item in dataset]


# ---------------------------------------------------------------------------
# Step 2: Ragas scoring (per metric, in batches, resumable)
# ---------------------------------------------------------------------------
def _seed_from_old_csv(scores: dict) -> dict:
    """One-time import of scores from the earlier all-at-once run's CSV."""
    if scores or not _REPORT_CSV.exists():
        return scores
    old = pd.read_csv(_REPORT_CSV)
    q_col = "user_input" if "user_input" in old.columns else "question"
    if q_col not in old.columns:
        return scores
    for _, row in old.iterrows():
        for col in _OLD_METRIC_COLS:
            if col in old.columns and not pd.isna(row[col]):
                scores.setdefault(row[q_col], {})[col] = float(row[col])
    return scores


def score_records(records: list[dict]) -> dict:
    judge_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(model=JUDGE_MODEL, temperature=0, thinking_level="minimal")
    )
    judge_emb = LangchainEmbeddingsWrapper(get_embeddings_client())
    run_config = RunConfig(timeout=600, max_retries=8, max_wait=60, max_workers=2)

    metrics = [
        LLMContextRecall(),
        LLMContextPrecisionWithReference(),
        Faithfulness(),
        ResponseRelevancy(strictness=1),  # n>1 completions unsupported on Gemma
    ]

    scores = _seed_from_old_csv(_load_cache(_SCORES_PATH))

    for metric in metrics:
        pending = [
            r for r in records
            if _is_missing(scores.get(r["sample"]["user_input"], {}).get(metric.name))
        ]
        print(f"\n{metric.name}: {len(pending)} of {len(records)} to score")

        for i in range(0, len(pending), SCORE_BATCH_SIZE):
            batch = pending[i : i + SCORE_BATCH_SIZE]
            result = evaluate(
                dataset=EvaluationDataset.from_list([r["sample"] for r in batch]),
                metrics=[metric],
                llm=judge_llm,
                embeddings=judge_emb,
                run_config=run_config,
                raise_exceptions=False,  # failed job -> NaN, retried next run
                show_progress=True,
            )
            df = result.to_pandas()
            col = metric.name if metric.name in df.columns else df.select_dtypes("number").columns[0]
            for r, v in zip(batch, df[col].tolist()):
                scores.setdefault(r["sample"]["user_input"], {})[metric.name] = (
                    None if _is_missing(v) else float(v)
                )
            _save_cache(scores, _SCORES_PATH)  # save after every batch

    return scores


# ---------------------------------------------------------------------------
# Step 3: timestamp metrics + report
# ---------------------------------------------------------------------------
def _overlaps(span, start: float, end: float) -> bool:
    return span[0] <= end and start <= span[1]


def build_report(dataset: list[dict], records: list[dict], scores: dict, transcript) -> pd.DataFrame:
    """
    One row per question: Ragas scores + timestamp metrics.

      ts_precision      fraction of the retrieved chunks overlapping the answer window
      ts_max_precision  best achievable ts_precision: chunks that overlap the window
                        (capped at top_k) / top_k. A short window can only touch a
                        couple of chunks, so this is the metric's ceiling.
      ts_hit            1 if any retrieved chunk overlaps the window (recall@k)
    """
    all_chunks = chunk_transcript(transcript, **CHUNK_KWARGS)
    rows = []
    for item, rec in zip(dataset, records):
        a, b = item["answer_start_sec"], item["answer_end_sec"]
        spans = rec["source_spans"]
        n_hit = sum(_overlaps(s, a, b) for s in spans)
        n_window = sum(_overlaps((c.start, c.end), a, b) for c in all_chunks)
        ts_prec = n_hit / len(spans) if spans else 0.0
        ts_max = min(n_window, TOP_K) / TOP_K
        rows.append({
            "question": item["question"],
            **scores.get(item["question"], {}),
            "ts_precision": ts_prec,
            "ts_max_precision": ts_max,
            "ts_precision_vs_max": ts_prec / ts_max if ts_max > 0 else float("nan"),
            "ts_hit": 1.0 if n_hit else 0.0,
        })
    return pd.DataFrame(rows)


def main(limit: int | None = None) -> None:
    transcript, dataset = load_eval_dataset()
    if limit:
        dataset = dataset[:limit]
    video_id = transcript.video_id

    if not is_video_indexed(video_id, COLLECTION_NAME):
        raise SystemExit(
            f"Video {video_id} is not in collection {COLLECTION_NAME!r}. Index it first."
        )

    records = generate_records(dataset, video_id)
    scores = score_records(records)
    df = build_report(dataset, records, scores, transcript)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(_REPORT_CSV, index=False)

    nums = df.select_dtypes("number")
    summary = pd.DataFrame({
        "mean": nums.mean().round(3),
        "valid": nums.count(),
        "nan": nums.isna().sum(),
    })
    print("\n=== Summary (mean is over valid rows only; nan = judge call failed) ===")
    print(summary.to_string())

    prec_col = next(
        (c for c in nums.columns if "precision" in c and not c.startswith("ts_")), None
    )
    if prec_col:
        print(f"\nCorrelation({prec_col}, ts_precision) = "
              f"{nums[prec_col].corr(nums['ts_precision']):.2f}")

    print(f"\nSaved per-question report -> {_REPORT_CSV}")
    if summary["nan"].any():
        print("Some scores are still NaN - just re-run to retry only those.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="only evaluate the first N questions")
    main(limit=parser.parse_args().limit)