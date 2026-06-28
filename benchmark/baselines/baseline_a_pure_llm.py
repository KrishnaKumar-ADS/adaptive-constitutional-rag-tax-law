"""
benchmark/baselines/baseline_a_pure_llm.py

Baseline A: Pure LLM (Groq gpt-oss-120b) without any retrieval context.
Serves as the bottom-line reference for hallucination rate.
"""
import asyncio
from src.llm.inference_groq import generate_groq_raw
from src.generation.output_schema import LegalResponse

async def run_baseline_a(question: str) -> LegalResponse:
    import time
    start = time.perf_counter()
    
    answer = await generate_groq_raw(question, temperature=0.1)
    
    elapsed_ms = (time.perf_counter() - start) * 1000
    
    return LegalResponse(
        question=question,
        answer=answer,
        verdict="N/A",  # No verification
        confidence=0.0,
        evidence_count=0,
        evidence=[],
        retrieval_method="none",
        model_name="groq/gpt-oss-120b",
        decision="Answered",
        uncertainty_score=0.0,
        strictness_tier="low",
        processing_time_ms=elapsed_ms,
        citations_verified=[]
    )
