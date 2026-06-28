"""
benchmark/metrics/legal_groundedness_score.py
"""
def calculate_lgs(hallucination_rate: float, citation_precision: float) -> float:
    """
    Legal Groundedness Score (LGS)
    A composite metric proposed for Day 14.
    LGS = (1 - Hallucination Rate) * Citation Precision
    """
    return (1.0 - hallucination_rate) * citation_precision
