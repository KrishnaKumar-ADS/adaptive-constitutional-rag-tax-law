"""
benchmark/baselines/__init__.py
"""
from .baseline_a_pure_llm import run_baseline_a
from .baseline_b_finetuned_llm import run_baseline_b
from .baseline_c_standard_rag import run_baseline_c
from .baseline_d_rag_static_rules import run_baseline_d
from .baseline_e_full_system import run_baseline_e

__all__ = [
    "run_baseline_a",
    "run_baseline_b",
    "run_baseline_c",
    "run_baseline_d",
    "run_baseline_e"
]
