import numpy as np

def retrieval_confidence(scores: list[float]) -> float:
    """How close the top retrieved documents are — mean of top-3 fused scores."""
    if not scores:
        return 0.0
    return float(np.mean(sorted(scores, reverse=True)[:3]))

def evidence_agreement(nli_entailment_scores: list[float]) -> float:
    """
    Do retrieved passages support the same conclusion? 
    Given NLI entailment scores (0-1) between the claim and each passage, 
    returns the mean agreement.
    """
    if not nli_entailment_scores:
        return 0.0
    return float(np.mean(nli_entailment_scores))

def coverage(query_terms: set[str], evidence_terms: set[str]) -> float:
    """Does retrieved evidence span the legal concepts in the query? Jaccard-style overlap."""
    if not query_terms:
        return 1.0
    return len(query_terms & evidence_terms) / max(len(query_terms), 1)

def entropy(scores: list[float]) -> float:
    """Ambiguity of the retrieval distribution — normalized Shannon entropy over softmaxed scores."""
    if not scores:
        return 0.0
    if len(scores) == 1:
        return 0.0
    scores = np.array(scores)
    # subtract max for numerical stability
    scores_shifted = scores - np.max(scores)
    p = np.exp(scores_shifted) / np.sum(np.exp(scores_shifted))
    # Add epsilon to prevent log(0)
    p = p + 1e-9
    return float(-np.sum(p * np.log(p)) / np.log(len(p)))
