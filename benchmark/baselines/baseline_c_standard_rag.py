"""Baseline C: Standard RAG pipeline.

Wires: hybrid_search() → evidence_aggregator → build_prompt →
       inference_openrouter() → citation_verifier() → response_builder()
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


def run_baseline_c(question: str):
    """
    End-to-end standard RAG:
      1. Hybrid retrieval (dense + sparse + RRF)
      2. Evidence aggregation & dedup
      3. Prompt construction with evidence
      4. LLM generation via OpenRouter
      5. Citation verification (existence + grounding + NLI)
      6. Structured response assembly
    """

    # Step 1: Hybrid retrieval
    retrieved = hybrid_search(
        question,
        top_k=10,
    )

    # Step 2: Aggregate & deduplicate evidence
    evidence_set = aggregate_evidence(
        retrieved,
    )

    # Step 3: Build prompt with evidence
    prompt = build_prompt(
        question,
        evidence_set,
    )

    # Step 4: Generate answer via OpenRouter
    answer = inference_openrouter(prompt)

    # Step 5: Verify citations in the answer
    verifier = CitationVerifier()

    citation_result = verifier.verify(
        claim=answer,
        evidence_list=evidence_set.evidences,
    )

    # Step 6: Build structured response
    response = build_response(
        question=question,
        answer=answer,
        evidence_set=evidence_set,
        citation_result=citation_result,
        model_name="openai/gpt-oss-120b:free",
    )

    return response