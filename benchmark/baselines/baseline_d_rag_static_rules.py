"""
benchmark/baselines/baseline_d_rag_static_rules.py

Baseline D: RAG with Static Strictness (always 'high') + Verification.
Uses the fine-tuned model (Local) and constitutional rules, but ignores uncertainty.
"""
import asyncio
from src.retrieval.hybrid_retriever import hybrid_search
from src.evidence.evidence_aggregator import aggregate_evidence
from src.llm.inference_local import generate_local
from src.citation.citation_verifier import CitationVerifier
from src.generation.output_schema import LegalResponse, EvidenceReference

# Lazy verifier
_verifier = None
def _get_verifier():
    global _verifier
    if _verifier is None:
        _verifier = CitationVerifier()
    return _verifier

async def run_baseline_d(question: str) -> LegalResponse:
    import time
    start = time.perf_counter()
    
    # 1. Retrieval
    results = hybrid_search(question, top_k=8)
    evidence_set = aggregate_evidence(results)
    
    flat_evidence = "\n\n".join(
        f"[Section {e.section_number or e.article_number or 'Unknown'}] {e.text}"
        for e in evidence_set.evidences
    )
    
    # 2. Generation (always high strictness)
    resp = await generate_local(question, flat_evidence, strictness_tier="high")
    answer = resp["text"]
    
    # 3. Verification
    verdict_result = _get_verifier().verify(answer, evidence_set.evidences)
    decision = "Answered with Hallucination" if verdict_result.get("verdict") == "Invalid" else "Answered"
    
    # Simple strictness logic: if high strictness abstains
    if answer.strip().startswith("I cannot answer"):
        decision = "Abstained"
        
    elapsed_ms = (time.perf_counter() - start) * 1000
    
    refs = [
        EvidenceReference(citation_id=e.citation_id, source_type=e.source_type, score=e.score)
        for e in evidence_set.evidences
    ]
    
    return LegalResponse(
        question=question,
        answer=answer,
        verdict=verdict_result.get("verdict", "Valid"),
        confidence=verdict_result.get("score", 0.0),
        evidence_count=len(refs),
        evidence=refs,
        retrieval_method="hybrid",
        model_name="qwen3-tax-local",
        decision=decision,
        uncertainty_score=0.0,  # Ignored in this baseline
        strictness_tier="high",
        processing_time_ms=elapsed_ms,
        citations_verified=[verdict_result]
    )
