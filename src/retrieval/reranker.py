"""Cross-encoder reranking using BGE-Reranker."""
from FlagEmbedding import FlagReranker

reranker = FlagReranker(
    "BAAI/bge-reranker-v2-m3",
    use_fp16=True
)


def rerank(
    query,
    passages,
    top_k=8
):
    pairs = [
        (query, p["text"])
        for p in passages
    ]

    scores = reranker.compute_score(
        pairs
    )

    ranked = sorted(
        zip(passages, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        p
        for p, _
        in ranked[:top_k]
    ]