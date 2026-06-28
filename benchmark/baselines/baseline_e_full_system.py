"""
benchmark/baselines/baseline_e_full_system.py

Baseline E: Adaptive Constitutional RAG.
The full system pipeline implemented in src/pipeline.py with async routing.
"""
import asyncio
from src.pipeline import run_pipeline_async
from src.generation.output_schema import LegalResponse

async def run_baseline_e(question: str) -> LegalResponse:
    # run_pipeline_async already returns the fully populated LegalResponse
    # Use local model for the full system
    return await run_pipeline_async(question, use_local_model=True)
