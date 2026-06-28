# src/generation/response_builder.py

from src.generation.output_schema import (
    LegalResponse,
    EvidenceReference,
)


def build_response(
    question: str,
    answer: str,
    evidence_set,
    citation_result,
    model_name="groq",
    quality_score=None,
    issues_found=None,
    reformatted=None,
):

    evidence_refs = []

    for ev in evidence_set.evidences:

        evidence_refs.append(
            EvidenceReference(
                citation_id=ev.citation_id,
                source_type=ev.source_type,
                score=round(ev.score, 4),
            )
        )

    return LegalResponse(
        question=question,
        answer=answer,
        verdict=citation_result["verdict"],
        confidence=round(
            citation_result.get("score", 0.0),
            4,
        ),
        evidence_count=len(
            evidence_set.evidences
        ),
        evidence=evidence_refs,
        retrieval_method="Hybrid-RRF",
        model_name=model_name,
        quality_score=quality_score,
        issues_found=issues_found,
        reformatted=reformatted,
    )

async def build_response_async(
    question: str,
    answer_text: str,
    evidence_set,
    citation_results: list[dict],
    model_name: str,
    decision: str = None,
    uncertainty_score: float = None,
    strictness_tier: str = None,
    processing_time_ms: float = None,
    quality_score: float = None,
    issues_found: list[str] = None,
    reformatted: bool = None,
) -> LegalResponse:
    """Async response builder that populates the full Day 13/14 schema."""
    evidence_refs = []
    
    for ev in evidence_set.evidences:
        evidence_refs.append(
            EvidenceReference(
                citation_id=ev.citation_id,
                source_type=ev.source_type,
                score=ev.score,
            )
        )
        
    # Aggregate verdicts: if any citation is Invalid, the whole response is Invalid
    final_verdict = "Valid"
    for res in citation_results:
        v = res.get("verdict", "Valid")
        if v == "Invalid":
            final_verdict = "Invalid"
            break
        elif v == "Partially Supported" and final_verdict != "Invalid":
            final_verdict = "Partially Supported"
            
    # Handle abstention
    if decision == "Abstained":
        final_verdict = "Abstained"

    confidence = citation_results[0].get("score", 0.0) if citation_results else 0.0

    return LegalResponse(
        question=question,
        answer=answer_text,
        verdict=final_verdict,
        confidence=confidence,
        evidence_count=len(evidence_refs),
        evidence=evidence_refs,
        retrieval_method="hybrid",
        model_name=model_name,
        decision=decision,
        uncertainty_score=uncertainty_score,
        strictness_tier=strictness_tier,
        processing_time_ms=processing_time_ms,
        citations_verified=citation_results,
        quality_score=quality_score,
        issues_found=issues_found,
        reformatted=reformatted,
    )