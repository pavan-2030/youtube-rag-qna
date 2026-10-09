from retriever import retrieve

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

def _format_timestamp(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes:02d}:{secs:02d}"

def is_hit(retrieved_chunk, answer_start, answer_end):
    """
    Return True if retrieved_chunk's time range overlaps
    [answer_start, answer_end].
    """
    chunk_start = retrieved_chunk.start
    chunk_end = retrieved_chunk.end
    # Question for you: what's the actual condition for two
    # time ranges [a,b] and [c,d] to overlap? Write it as a
    # single boolean expression, not a bunch of if/else.
    return chunk_start <= answer_end and answer_start <= chunk_end

def evaluate(ground_truth, k=3, video_id=None):
    recalls = []
    precisions = []

    for item in ground_truth:
        retrieved = retrieve(item["question"], top_k=k, video_id=video_id)
        # blocks = []
        # for c in retrieved:
        #     ts = f"[{_format_timestamp(c.start)}-{_format_timestamp(c.end)}]"
        #     blocks.append(f"{ts} {c.text}")

        hits = [c for c in retrieved if is_hit(c, item["answer_start_sec"], item["answer_end_sec"])]

        # Question for you: given `hits` and `retrieved`, and knowing
        # there's exactly ONE relevant time-window per question here,
        # how do you compute recall and precision for this single question?
        recall = 1 if len(hits) > 0 else 0  # since there's exactly one relevant time-window
        precision = len(hits) / len(retrieved) if retrieved else 0

        recalls.append(recall)
        precisions.append(precision)

    avg_recall = sum(recalls) / len(recalls)
    avg_precision = sum(precisions) / len(precisions)
    return avg_recall, avg_precision

if __name__ == "__main__":
    recall, precision = evaluate(ground_truth, k=3, video_id="8idr1WZ1A7Q")
    print(f"Recall@3: {recall:.2f}")
    print(f"Precision@3: {precision:.2f}")