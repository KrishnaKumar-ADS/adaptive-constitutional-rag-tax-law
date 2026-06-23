"""Main pipeline orchestrating all layers of the system.

Day 7 deliverable: End-to-end query → structured response.

Layers used:
  Layer 1: Query processing
  Layer 2: Hybrid retrieval (dense + sparse + RRF)
  Layer 3: Evidence aggregation
  Layer 5: LLM generation (OpenRouter free model)
  Layer 7: Citation verification
  Layer 8: Structured output
"""

from src.retrieval.hybrid_retriever import (
    hybrid_search,
)

from src.evidence.evidence_aggregator import (
    aggregate_evidence,
)

from src.llm.prompts import (
    build_prompt,
)

from src.llm.openrouter import (
    inference_openrouter,
)

from src.citation.citation_verifier import (
    CitationVerifier,
)

from src.generation.response_builder import (
    build_response,
)

from src.db.crud import log_query


# Lazy-loaded verifier singleton
_verifier = None


def _get_verifier():
    global _verifier
    if _verifier is None:
        _verifier = CitationVerifier()
    return _verifier


def run_pipeline(
    question: str,
    top_k: int = 8,
    log_to_db: bool = True,
):
    """
    Full end-to-end pipeline:
      query → hybrid_retriever → evidence_aggregator
      → build_prompt → inference_openrouter
      → citation_verifier → response_builder
      → (optional) log to Postgres
    """

    # Layer 2: Hybrid retrieval
    retrieved = hybrid_search(
        question,
        top_k=top_k,
    )

    # Layer 3: Evidence aggregation
    evidence_set = aggregate_evidence(
        retrieved,
    )

    # Build prompt with evidence
    prompt = build_prompt(
        question,
        evidence_set,
    )

    # Layer 6: LLM generation
    answer = inference_openrouter(prompt)

    # Layer 7: Citation verification
    verifier = _get_verifier()

    citation_result = verifier.verify(
        claim=answer,
        evidence_list=evidence_set.evidences,
    )

    # Layer 8: Structured response
    response = build_response(
        question=question,
        answer=answer,
        evidence_set=evidence_set,
        citation_result=citation_result,
        model_name="openai/gpt-oss-120b:free",
    )

    # Log to Postgres (Day 7 requirement)
    if log_to_db:
        try:
            evidence_ids = [
                ev.citation_id
                for ev in evidence_set.evidences
            ]

            log_query(
                question=question,
                evidence_ids=evidence_ids,
                decision=citation_result.get("verdict"),
                citations_json=citation_result,
                raw_response=answer,
            )
        except Exception as e:
            print(f"Warning: DB logging failed: {e}")

    return response
