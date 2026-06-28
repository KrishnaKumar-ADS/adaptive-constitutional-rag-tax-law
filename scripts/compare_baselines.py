"""
scripts/compare_baselines.py

Runner script for the 5 comparative baselines (Day 13).
Executes A-E sequentially for a single question or small batch.
For the full evaluation sweep, see benchmark/run_evaluation.py (Day 14).
"""
import asyncio
import json
from benchmark.baselines import (
    run_baseline_a,
    run_baseline_b,
    run_baseline_c,
    run_baseline_d,
    run_baseline_e
)

async def compare_single(question: str):
    print(f"\nEvaluating Question: '{question}'\n" + "="*80)
    
    baselines = {
        "A (Pure LLM)": run_baseline_a,
        "B (Fine-tuned LLM)": run_baseline_b,
        "C (Standard RAG)": run_baseline_c,
        "D (RAG + Static Rules)": run_baseline_d,
        "E (Adaptive System)": run_baseline_e
    }
    
    results = {}
    for name, func in baselines.items():
        print(f"Running Baseline {name}...")
        try:
            resp = await func(question)
            results[name] = resp.model_dump()
            print(f"  Decision: {resp.decision}")
            print(f"  Verdict: {resp.verdict}")
            print(f"  Uncertainty: {resp.uncertainty_score:.3f}")
            print(f"  Time: {resp.processing_time_ms:.1f}ms")
        except Exception as e:
            print(f"  Failed: {e}")
            results[name] = {"error": str(e)}
            
    with open("benchmark/baseline_comparison_latest.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to benchmark/baseline_comparison_latest.json")

if __name__ == "__main__":
    q = "What is the penalty under Section 271F of the Income Tax Act for late filing?"
    asyncio.run(compare_single(q))
