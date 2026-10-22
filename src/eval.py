from chunker import chunk_transcript, load_transcript_json
from embedder import DEFAULT_COLLECTION_NAME, embed_and_store_chunks, get_qdrant_client
from retriever import retrieve
from matplotlib import pyplot as plt
from eval_data import ground_truth, TRANSCRIPT_PATH


CONFIGS = [(500, 100), (1000, 200), (1500, 300), (2000, 400)]  # (chunk_size, overlap)
KS = (1, 3, 5)


def is_hit(retrieved_chunk, answer_start, answer_end):
    """
    Return True if retrieved_chunk's time range overlaps
    [answer_start, answer_end].
    """
    chunk_start = retrieved_chunk.start
    chunk_end = retrieved_chunk.end
    return chunk_start <= answer_end and answer_start <= chunk_end

def evaluate(ground_truth, ks=KS, video_id=None, collection_name=DEFAULT_COLLECTION_NAME, chunks=None):
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
        n_window = sum(is_hit(c, item["answer_start_sec"], item["answer_end_sec"]) for c in chunks)
        for k in ks:
            top = retrieved[:k]
            hits = [c for c in top if is_hit(c, item["answer_start_sec"], item["answer_end_sec"])]
            recalls[k].append(1 if hits else 0)
            ceiling = min(k, n_window)
            precisions[k].append(len(hits) / ceiling if ceiling else 0)

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
    scores = evaluate(ground_truth, ks=ks, video_id=transcript.video_id, collection_name=collection, chunks=chunks)
    avg_len = sum(len(c.text) for c in chunks) / len(chunks)
    return scores, len(chunks), avg_len

if __name__ == "__main__":
    transcript = load_transcript_json(TRANSCRIPT_PATH)

    results = {}

    print(f"{'size/overlap':<14}{'chunks':>7}{'avg_len':>9}  " +
          "  ".join(f"R@{k}   P@{k}" for k in KS))
    for size, overlap in CONFIGS:
        results[(size, overlap)] = evaluate_chunk_config(transcript, ground_truth, size, overlap)
        scores, n_chunks, avg_len = results[(size, overlap)]
        row = "  ".join(f"{scores[k][0]:.2f}  {scores[k][1]:.2f}" for k in KS)
        print(f"{size}/{overlap:<10}{n_chunks:>7}{avg_len:>9.0f}  {row}")


    plt.figure(figsize=(10, 6))
    for size, overlap in CONFIGS:
        scores, _, _ = results[(size, overlap)]
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