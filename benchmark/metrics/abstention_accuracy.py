"""
benchmark/metrics/abstention_accuracy.py
"""
def calculate_abstention_accuracy(results: list[dict]) -> float:
    """
    Abstention Accuracy
    Measures how often the model correctly abstained when it should have 
    (i.e. when answering would have resulted in an Invalid verdict).
    Note: Requires a counterfactual "what would it have answered" which we approximate
    in the baselines by looking at standard RAG failures vs Adaptive RAG abstentions.
    For standard metric, we calculate % of total questions abstained.
    """
    abstained = sum(1 for r in results if r.get("decision") == "Abstained")
    return abstained / len(results) if results else 0.0
