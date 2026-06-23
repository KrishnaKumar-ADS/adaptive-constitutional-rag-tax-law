"""
Adaptive Retrieval Layer

Day 6:
- Confidence estimation
- Adaptive expansion
- Confidence labeling
- Hallucination prevention support
"""

from src.retrieval.hybrid_retriever import hybrid_search


def calculate_confidence(
    query: str,
    results: list
) -> float:
    """
    Confidence based on:

    1. Retrieval score
    2. Query term coverage

    Final score:
    70% retrieval
    30% keyword coverage
    """

    if not results:
        return 0.0

    # -------------------------
    # Retrieval confidence
    # -------------------------

    scores = [r.score for r in results[:3]]

    while len(scores) < 3:
        scores.append(0.0)

    retrieval_confidence = (
        scores[0] * 0.5 +
        scores[1] * 0.3 +
        scores[2] * 0.2
    )

    # -------------------------
    # Keyword coverage
    # -------------------------

    query_words = {
        w.lower()
        for w in query.split()
        if len(w) > 3
    }

    text = " ".join(
        r.payload["text"].lower()
        for r in results[:3]
    )

    matched = sum(
        1
        for word in query_words
        if word in text
    )

    coverage = matched / max(len(query_words), 1)

    confidence = (
        retrieval_confidence * 0.7 +
        coverage * 0.3
    )

    return round(confidence, 4)


def get_confidence_label(confidence: float):

    if confidence >= 0.70:
        return "high"

    if confidence >= 0.50:
        return "medium"

    if confidence >= 0.30:
        return "low"

    return "very_low"


def needs_expansion(
    confidence: float,
    threshold: float = 0.50
) -> bool:

    return confidence < threshold


def adaptive_search(
    query: str,
    threshold: float = 0.50
):

    results = hybrid_search(
        query=query,
        top_k=8
    )

    confidence = calculate_confidence(
        query,
        results
    )

    retrieval_mode = "normal"

    if needs_expansion(
        confidence,
        threshold
    ):

        results = hybrid_search(
            query=query,
            top_k=15
        )

        retrieval_mode = "expanded"

        confidence = calculate_confidence(
            query,
            results
        )

    return {
        "results": results,
        "confidence": confidence,
        "confidence_label": get_confidence_label(
            confidence
        ),
        "retrieval_mode": retrieval_mode,
        "abstain": should_abstain(
            confidence
        )
    }


def should_abstain(
    confidence: float,
    abstain_threshold: float = 0.45
):
    return confidence < abstain_threshold