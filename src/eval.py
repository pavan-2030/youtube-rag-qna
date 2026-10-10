from chunker import chunk_transcript, load_transcript_json
from embedder import DEFAULT_COLLECTION_NAME, embed_and_store_chunks, get_qdrant_client
from retriever import retrieve
from matplotlib import pyplot as plt

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

TRANSCRIPT_PATH = "data/transcripts/8idr1WZ1A7Q.json"
CONFIGS = [(500, 100), (1000, 200), (1500, 300), (2000, 400)]  # (chunk_size, overlap)
KS = (1, 3, 5)

# def _format_timestamp(seconds: float) -> str:
#     minutes, secs = divmod(int(seconds), 60)
#     return f"{minutes:02d}:{secs:02d}"

def is_hit(retrieved_chunk, answer_start, answer_end):
    """
    Return True if retrieved_chunk's time range overlaps
    [answer_start, answer_end].
    """
    chunk_start = retrieved_chunk.start
    chunk_end = retrieved_chunk.end
    return chunk_start <= answer_end and answer_start <= chunk_end

def evaluate(ground_truth, ks=KS, video_id=None, collection_name=DEFAULT_COLLECTION_NAME):
    """Return {k: (recall@k, precision@k)} for one collection.

    Retrieves once at max(ks) and slices, since results come back ranked by score.
    """
    k_max = max(ks)
    recalls = {k: [] for k in ks}
    precisions = {k: [] for k in ks}

    for item in ground_truth:
        retrieved = retrieve(
            item["question"],
            top_k=k_max,
            video_id=video_id,
            collection_name=collection_name,
        )
        for k in ks:
            top = retrieved[:k]
            hits = [c for c in top if is_hit(c, item["answer_start_sec"], item["answer_end_sec"])]
            recalls[k].append(1 if hits else 0)
            precisions[k].append(len(hits) / len(top) if top else 0)

    return {k: (sum(recalls[k]) / len(recalls[k]), sum(precisions[k]) / len(precisions[k])) for k in ks}

def build_eval_collection(transcript, chunk_size, overlap, collection_name):
    """Fresh collection per config, so old chunks never mix with new ones."""
    client = get_qdrant_client()
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    chunks = chunk_transcript(
        transcript,
        chunk_size_chars=chunk_size,
        chunk_overlap_chars=overlap,
    )
    embed_and_store_chunks(chunks, collection_name=collection_name)
    return chunks

def evaluate_chunk_config(transcript, ground_truth, chunk_size, overlap, ks=KS):
    collection = f"eval_{chunk_size}_{overlap}"
    chunks = build_eval_collection(transcript, chunk_size, overlap, collection)
    scores = evaluate(ground_truth, ks=ks, video_id=transcript.video_id, collection_name=collection)
    avg_len = sum(len(c.text) for c in chunks) / len(chunks)
    return scores, len(chunks), avg_len

if __name__ == "__main__":
    transcript = load_transcript_json(TRANSCRIPT_PATH)

    print(f"{'size/overlap':<14}{'chunks':>7}{'avg_len':>9}  " +
          "  ".join(f"R@{k}   P@{k}" for k in KS))
    for size, overlap in CONFIGS:
        scores, n_chunks, avg_len = evaluate_chunk_config(transcript, ground_truth, size, overlap)
        row = "  ".join(f"{scores[k][0]:.2f}  {scores[k][1]:.2f}" for k in KS)
        print(f"{size}/{overlap:<10}{n_chunks:>7}{avg_len:>9.0f}  {row}")


    plt.figure(figsize=(10, 6))
    for size, overlap in CONFIGS:
        scores, n_chunks, avg_len = evaluate_chunk_config(transcript, ground_truth, size, overlap)
        plt.plot(KS, [scores[k][0] for k in KS], marker='o', label=f"R@k {size}/{overlap}")
        plt.plot(KS, [scores[k][1] for k in KS], marker='x', label=f"P@k {size}/{overlap}")

    plt.title("Recall and Precision at k for Different Chunk Configurations")
    plt.xlabel("k")
    plt.ylabel("Score")
    plt.xticks(KS)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("eval_results.png")
    plt.show()