"""Complete citation verification pipeline.

Orchestrates:
  1. Section/Article existence check (ground-truth index lookup)
  2. Grounding check (was the citation actually retrieved?)
  3. NLI relevance check (does the evidence entail the claim?)

Returns: Valid / Invalid / Partially Supported
"""
from src.citation.section_validator import (
    SectionValidator,
)
from src.citation.article_validator import (
    ArticleValidator,
)
from src.citation.grounding_checker import (
    GroundingChecker,
)
from src.citation.relevance_checker import (
    RelevanceChecker,
)


class CitationVerifier:

    def __init__(self):
        self.section_validator = (
            SectionValidator()
        )
        self.article_validator = (
            ArticleValidator()
        )
        self.grounding_checker = (
            GroundingChecker()
        )
        self.relevance_checker = (
            RelevanceChecker()
        )

    def verify(
        self,
        claim: str,
        evidence_list,
    ):
        """
        Verify citations in a claim against evidence.

        Args:
            claim: The generated answer text containing citations
            evidence_list: List of Evidence objects (from EvidenceSet.evidences)
                          or list of dicts with section_number/article_number/text keys
        """

        # Step 1: Extract and validate section/article references
        section_results = (
            self.section_validator.validate(claim)
        )
        article_results = (
            self.article_validator.validate(claim)
        )

        citations = section_results + article_results

        if not citations:
            return {
                "verdict": "Invalid",
                "reason": "No citations found in claim",
                "score": 0.0,
            }

        # Step 2: Check existence in ground-truth index
        for citation in citations:
            exists = citation.get("exists", False)

            if not exists:
                return {
                    "verdict": "Invalid",
                    "reason": f"Citation does not exist: {citation}",
                    "citation": citation,
                    "score": 0.0,
                }

        # Step 3: Check grounding (was citation actually retrieved?)
        for citation in citations:
            ref = (
                citation.get("section")
                or citation.get("article")
            )

            grounded = (
                self.grounding_checker.citation_used(
                    ref,
                    evidence_list,
                )
            )

            if not grounded:
                return {
                    "verdict": "Partially Supported",
                    "reason": f"Citation {ref} exists but was not in retrieved evidence",
                    "citation": ref,
                    "score": 0.3,
                }

        # Step 4: NLI entailment check against best evidence
        best_evidence_text = ""
        if evidence_list:
            first = evidence_list[0]
            if hasattr(first, "text"):
                best_evidence_text = first.text
            elif isinstance(first, dict):
                best_evidence_text = first.get("text", "")

        if not best_evidence_text:
            return {
                "verdict": "Partially Supported",
                "reason": "No evidence text available for entailment check",
                "score": 0.3,
            }

        result = self.relevance_checker.entails(
            claim,
            best_evidence_text,
        )

        if result["entails"]:
            return {
                "verdict": "Valid",
                "score": result["score"],
            }

        return {
            "verdict": "Partially Supported",
            "reason": "Evidence does not strongly entail claim",
            "score": result["score"],
        }