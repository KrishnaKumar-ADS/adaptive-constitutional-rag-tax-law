"""NLI entailment check using cross-encoder.

Guide Day 6 specifies: cross-encoder/nli-deberta-v3-base
This is a proper NLI model, not a bi-encoder similarity model.
"""
from sentence_transformers import CrossEncoder


class RelevanceChecker:

    def __init__(self):
        self.model = CrossEncoder(
            "cross-encoder/nli-deberta-v3-base"
        )

    def entails(
        self,
        claim: str,
        evidence: str,
        threshold: float = 0.5,
    ):
        """
        NLI cross-encoder returns scores for
        [contradiction, entailment, neutral].
        We use the entailment score as our relevance signal.
        """
        scores = self.model.predict(
            [(evidence, claim)]
        )

        # nli-deberta-v3-base outputs 3 logits:
        # [contradiction, entailment, neutral]
        if hasattr(scores[0], "__len__"):
            # Multi-class output
            import numpy as np
            probs = np.exp(scores[0]) / np.sum(np.exp(scores[0]))
            entailment_score = float(probs[1])  # entailment index
        else:
            # Single score output (some cross-encoder versions)
            entailment_score = float(scores[0])

        return {
            "entails": entailment_score >= threshold,
            "score": entailment_score,
        }