"""
benchmark/baselines/baseline_b_finetuned_llm.py

Baseline B: Fine-tuned Qwen3-8B-Tax (Local Ollama) without retrieval context.
Measures how much domain knowledge the model internalized during QLoRA.
"""
import asyncio
from src.llm.inference_local import generate_local_raw
from src.generation.output_schema import LegalResponse

async def run_baseline_b(question: str) -> LegalResponse:
    import time
    start = time.perf_counter()
    
    answer = await generate_local_raw(question, temperature=0.1)
    
    elapsed_ms = (time.perf_counter() - start) * 1000
    
    return LegalResponse(
        question=question,
        answer=answer,
        verdict="N/A",
        confidence=0.0,
        evidence_count=0,
        evidence=[],
        retrieval_method="none",
        model_name="qwen3-tax-local",
        decision="Answered",
        uncertainty_score=0.0,
        strictness_tier="low",
        processing_time_ms=elapsed_ms,
        citations_verified=[]
    )
