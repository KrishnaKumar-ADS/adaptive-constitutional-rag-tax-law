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
    model_name="openrouter",
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
    )