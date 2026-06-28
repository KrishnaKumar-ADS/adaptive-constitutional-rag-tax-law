"""Complete citation verification pipeline.

Orchestrates:
  1. Section/Article existence check (ground-truth index lookup)
  2. Grounding check (was the citation actually retrieved?)
  3. NLI relevance check (does the evidence entail the claim?)

Returns: Valid / Invalid / Partially Supported

Collects ALL violations instead of failing fast on the first one,
so the pipeline can report and strip every hallucinated citation.
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
        Verify ALL citations in a claim against evidence.

        Unlike the previous fail-fast approach, this collects every violation
        so the pipeline can report (and strip) all hallucinated citations.

        Args:
            claim: The generated answer text containing citations
            evidence_list: List of Evidence objects (from EvidenceSet.evidences)
                          or list of dicts with section_number/article_number/text keys

        Returns:
            dict with:
              - verdict: "Valid" | "Partially Supported" | "Invalid"
              - score: float 0.0-1.0
              - reason: str
              - all_violations: list of dicts describing each violation
              - nonexistent_citations: list of citation refs that don't exist in the Act
              - ungrounded_citations: list of citation refs not in retrieved evidence
        """

        # Step 1: Extract and validate section/article references
        section_results = self.section_validator.validate(claim)
        article_results = self.article_validator.validate(claim)

        citations = section_results + article_results

        if not citations:
            return {
                "verdict": "Invalid",
                "reason": "No citations found in claim",
                "score": 0.0,
                "all_violations": [{"type": "no_citations", "detail": "Answer contains no legal citations"}],
                "nonexistent_citations": [],
                "ungrounded_citations": [],
            }

        # Collect ALL violations instead of stopping at first
        all_violations = []
        nonexistent_citations = []
        ungrounded_citations = []
        valid_count = 0

        # Step 2 + 3: Check existence AND grounding for every citation
        for citation in citations:
            exists = citation.get("exists", False)
            ref = citation.get("section") or citation.get("article")

            if not exists:
                nonexistent_citations.append(ref)
                all_violations.append({
                    "type": "nonexistent",
                    "citation": ref,
                    "detail": f"Citation '{ref}' does not exist in the Income Tax Act / Constitution",
                })
                continue

            # Check grounding (was citation actually retrieved?)
            grounded = self.grounding_checker.citation_used(ref, evidence_list)

            if not grounded:
                ungrounded_citations.append(ref)
                all_violations.append({
                    "type": "ungrounded",
                    "citation": ref,
                    "detail": f"Citation '{ref}' exists but was NOT in retrieved evidence",
                })
            else:
                valid_count += 1

        # Determine overall verdict based on violation severity
        total_citations = len(citations)

        if nonexistent_citations:
            # Any fabricated citation -> Invalid
            verdict = "Invalid"
            score = 0.0
            reason = (
                f"Found {len(nonexistent_citations)} nonexistent citation(s): "
                f"{', '.join(nonexistent_citations)}"
            )
        elif ungrounded_citations:
            if valid_count == 0:
                verdict = "Invalid"
                score = 0.1
                reason = (
                    f"All {len(ungrounded_citations)} citation(s) exist but "
                    f"none were in retrieved evidence: "
                    f"{', '.join(ungrounded_citations)}"
                )
            else:
                verdict = "Partially Supported"
                score = round(valid_count / total_citations, 2)
                reason = (
                    f"{len(ungrounded_citations)} citation(s) not in evidence: "
                    f"{', '.join(ungrounded_citations)}; "
                    f"{valid_count}/{total_citations} grounded"
                )
        else:
            # All citations exist and are grounded - run NLI entailment
            if not evidence_list:
                verdict = "Partially Supported"
                score = 0.3
                reason = "No evidence text available for entailment check"
            else:
                best_score = 0.0
                is_entailed = False
                
                # Check claim against all evidence chunks, since the relevant section
                # might not be the top-ranked (first) chunk.
                for evidence in evidence_list:
                    text = evidence.text if hasattr(evidence, "text") else evidence.get("text", "")
                    if not text:
                        continue
                        
                    result = self.relevance_checker.entails(claim, text)
                    if result["score"] > best_score:
                        best_score = result["score"]
                        
                    if result["entails"]:
                        is_entailed = True
                        break # Found an entailment, no need to check further

                if is_entailed:
                    verdict = "Valid"
                    score = best_score
                    reason = "All citations grounded and entailed by evidence"
                else:
                    verdict = "Partially Supported"
                    score = best_score
                    reason = "Citations grounded but evidence does not strongly entail claim"

        return {
            "verdict": verdict,
            "score": score,
            "reason": reason,
            "all_violations": all_violations,
            "nonexistent_citations": nonexistent_citations,
            "ungrounded_citations": ungrounded_citations,
        }