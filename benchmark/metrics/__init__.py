"""
benchmark/metrics/__init__.py
"""
from .hallucination_rate import calculate_hallucination_rate
from .citation_precision_recall import calculate_citation_metrics
from .calibration_error import calculate_ece
from .abstention_accuracy import calculate_abstention_accuracy
from .legal_groundedness_score import calculate_lgs

__all__ = [
    "calculate_hallucination_rate",
    "calculate_citation_metrics",
    "calculate_ece",
    "calculate_abstention_accuracy",
    "calculate_lgs"
]
