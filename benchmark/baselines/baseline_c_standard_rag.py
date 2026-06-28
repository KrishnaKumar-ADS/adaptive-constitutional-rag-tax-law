"""
benchmark/baselines/baseline_c_standard_rag.py

Baseline C: Standard RAG (Hybrid Search + Groq LLM).
No uncertainty quantification, no verification, no constitutional rules.
"""
import asyncio
from src.retrieval.hybrid_retriever import hybrid_search
from src.evidence.evidence_aggregator import aggregate_evidence
from src.llm.inference_groq import generate_groq
from src.generation.output_schema import LegalResponse, EvidenceReference

async def run_baseline_c(question: str) -> LegalResponse:
    import time
    start = time.perf_counter()
    
    # 1. Retrieval
    results = hybrid_search(question, top_k=8)
    evidence_set = aggregate_evidence(results)
    
    flat_evidence = "\n\n".join(
        f"[Section {e.section_number or e.article_number or 'Unknown'}] {e.text}"
        for e in evidence_set.evidences
    )
    
    # 2. Generation
    # Uses standard 'low' strictness, meaning it just answers normally.
    resp = await generate_groq(question, flat_evidence, strictness_tier="low")
    answer = resp["text"]
    
    elapsed_ms = (time.perf_counter() - start) * 1000
    
    refs = [
        EvidenceReference(citation_id=e.citation_id, source_type=e.source_type, score=e.score)
        for e in evidence_set.evidences
    ]
    
    return LegalResponse(
        question=question,
        answer=answer,
        verdict="N/A",  # Not verified
        confidence=0.0,
        evidence_count=len(refs),
        evidence=refs,
        retrieval_method="hybrid",
        model_name="groq/gpt-oss-120b",
        decision="Answered",
        uncertainty_score=0.0,
        strictness_tier="low",
        processing_time_ms=elapsed_ms,
        citations_verified=[]
    )