"""
src/llm/base_model_loader.py

Unified inference router — abstracts the choice between the local Ollama model
and the remote Groq API behind a single ``generate()`` call.
"""

from enum import Enum
from typing import Optional


class InferenceBackend(str, Enum):
    LOCAL = "local"        # Ollama + qwen3-tax GGUF
    GROQ = "groq"          # Groq API (gpt-oss-120b)


async def generate(
    question: str,
    evidence_text: str = "",
    strictness_tier: str = "medium",
    backend: InferenceBackend = InferenceBackend.LOCAL,
    temperature: float = 0.1,
    max_tokens: int = 1024,
) -> dict:
    """
    Route generation to the appropriate backend.

    Returns:
        dict with at minimum: {'text': str, 'model': str}
    """
    if backend == InferenceBackend.LOCAL:
        from src.llm.inference_local import generate_local
        return await generate_local(
            question=question,
            evidence_text=evidence_text,
            strictness_tier=strictness_tier,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    elif backend == InferenceBackend.GROQ:
        from src.llm.inference_groq import generate_groq
        return await generate_groq(
            question=question,
            evidence_text=evidence_text,
            strictness_tier=strictness_tier,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    else:
        raise ValueError(f"Unknown backend: {backend}")


def generate_sync(
    question: str,
    evidence_text: str = "",
    strictness_tier: str = "medium",
    backend: InferenceBackend = InferenceBackend.LOCAL,
    temperature: float = 0.1,
    max_tokens: int = 1024,
) -> dict:
    """
    Synchronous version of generate() — for backward compatibility
    and non-async contexts.
    """
    if backend == InferenceBackend.LOCAL:
        from src.llm.inference_local import generate_local_sync
        return generate_local_sync(
            question=question,
            evidence_text=evidence_text,
            strictness_tier=strictness_tier,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    elif backend == InferenceBackend.GROQ:
        from src.llm.openrouter import inference_groq
        from src.llm.prompts import build_evidence_prompt
        prompt = build_evidence_prompt(question, evidence_text)
        raw = inference_groq(prompt)
        return {
            "text": raw,
            "model": "groq",
        }

    else:
        raise ValueError(f"Unknown backend: {backend}")


def get_default_backend() -> InferenceBackend:
    """
    Auto-detect the best available backend.
    Prefers local if Ollama is healthy, falls back to Groq.
    """
    from src.llm.inference_local import check_ollama_health

    health = check_ollama_health()
    if health.get("model_loaded"):
        return InferenceBackend.LOCAL

    return InferenceBackend.GROQ
