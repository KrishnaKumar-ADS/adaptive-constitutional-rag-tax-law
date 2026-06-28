"""
src/llm/inference_local.py

Ollama-based local inference client for the fine-tuned Qwen3-8B-Tax GGUF model.
Uses the OpenAI-compatible API that Ollama exposes at /v1/chat/completions.
"""

import httpx
import json
import os
import time
from typing import Optional


OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
LOCAL_MODEL = os.environ.get("LOCAL_MODEL_NAME", "qwen3-tax")
DEFAULT_TIMEOUT = 600.0  # seconds — GGUF on CPU can be very slow for large RAG contexts


SYSTEM_PROMPT = (
    "You are a specialized Indian tax law assistant. "
    "ABSOLUTE RULES — VIOLATION IS UNACCEPTABLE:\n"
    "1. ONLY answer based on the legal provisions provided in the context below.\n"
    "2. You may ONLY cite Section/Article numbers that appear VERBATIM in the context text. "
    "NEVER invent, fabricate, or guess any Section or Article number.\n"
    "3. If the context does not contain enough information to fully answer the question, "
    "respond ONLY with: 'I cannot answer this based on available provisions.' — do NOT attempt a partial answer.\n"
    "4. NEVER add procedural details, cross-references, caveats, or exceptions from your own knowledge.\n"
    "5. Every claim must be directly supported by a specific passage in the context.\n"
    "6. Keep your answer concise: [Direct Answer] → [Legal Basis with exact citations from context] → "
    "[Caveats ONLY if stated in context].\n"
    "7. If you are unsure, ABSTAIN. Do not guess."
)


def build_chatml_prompt(
    question: str,
    evidence_text: str = "",
    strictness_tier: str = "medium",
) -> list[dict]:
    """
    Build a ChatML message list suitable for Ollama's /api/chat endpoint.

    Args:
        question: The user's legal query
        evidence_text: Concatenated evidence passages from retrieval
        strictness_tier: 'low', 'medium', or 'high' — adjusts the system prompt
    """
    # Tier-specific instructions
    tier_instructions = {
        "low": (
            "\nYou may provide a detailed response with reasoning. "
            "Cite all relevant sections/articles."
        ),
        "medium": (
            "\nAnswer with caution. Clearly flag any conflicting provisions. "
            "If evidence is ambiguous, note the uncertainty. "
            "Cite all sections/articles used."
        ),
        "high": (
            "\nINSUFFICIENT EVIDENCE DETECTED. You MUST abstain from answering. "
            "Respond ONLY with: 'I cannot answer this based on available provisions.' "
            "Do NOT attempt to reason or guess."
        ),
    }

    system_content = SYSTEM_PROMPT + tier_instructions.get(strictness_tier, tier_instructions["medium"])

    user_content = question
    if evidence_text:
        user_content = (
            f"Context (retrieved legal provisions):\n{evidence_text}\n\n"
            f"Question: {question}"
        )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


async def generate_local(
    question: str,
    evidence_text: str = "",
    strictness_tier: str = "medium",
    temperature: float = 0.1,
    max_tokens: int = 1024,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict:
    """
    Async call to Ollama's chat API.

    Returns:
        dict with keys: 'text', 'model', 'total_duration_ms', 'eval_count'
    """
    messages = build_chatml_prompt(question, evidence_text, strictness_tier)

    payload = {
        "model": LOCAL_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
        },
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        start = time.perf_counter()
        resp = await client.post(f"{OLLAMA_HOST}/api/chat", json=payload)
        elapsed_ms = (time.perf_counter() - start) * 1000

        resp.raise_for_status()
        data = resp.json()

    return {
        "text": data.get("message", {}).get("content", ""),
        "model": data.get("model", LOCAL_MODEL),
        "total_duration_ms": data.get("total_duration", 0) / 1e6,  # ns → ms
        "eval_count": data.get("eval_count", 0),
        "wall_time_ms": elapsed_ms,
    }


def generate_local_sync(
    question: str,
    evidence_text: str = "",
    strictness_tier: str = "medium",
    temperature: float = 0.1,
    max_tokens: int = 1024,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict:
    """
    Synchronous call to Ollama's chat API.

    Returns:
        dict with keys: 'text', 'model', 'total_duration_ms', 'eval_count'
    """
    messages = build_chatml_prompt(question, evidence_text, strictness_tier)

    payload = {
        "model": LOCAL_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
        },
    }

    start = time.perf_counter()
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(f"{OLLAMA_HOST}/api/chat", json=payload)
        elapsed_ms = (time.perf_counter() - start) * 1000

    resp.raise_for_status()
    data = resp.json()

    return {
        "text": data.get("message", {}).get("content", ""),
        "model": data.get("model", LOCAL_MODEL),
        "total_duration_ms": data.get("total_duration", 0) / 1e6,
        "eval_count": data.get("eval_count", 0),
        "wall_time_ms": elapsed_ms,
    }


def check_ollama_health() -> dict:
    """
    Check if Ollama is running and the model is loaded.

    Returns:
        dict with 'healthy' (bool) and 'models' (list) or 'error' (str)
    """
    try:
        with httpx.Client(timeout=5.0) as client:
            # Check Ollama is alive
            resp = client.get(f"{OLLAMA_HOST}/api/tags")
            resp.raise_for_status()
            data = resp.json()

        models = [m.get("name", "") for m in data.get("models", [])]
        model_loaded = any(LOCAL_MODEL in m for m in models)

        return {
            "healthy": True,
            "ollama_up": True,
            "model_loaded": model_loaded,
            "model_name": LOCAL_MODEL,
            "available_models": models,
        }
    except Exception as e:
        return {
            "healthy": False,
            "ollama_up": False,
            "model_loaded": False,
            "error": str(e),
        }


async def generate_local_raw(
    question: str,
    temperature: float = 0.1,
    max_tokens: int = 1024,
) -> str:
    """
    Raw generation without evidence injection — used by Baseline B.
    Returns just the text string.
    """
    result = await generate_local(
        question=question,
        evidence_text="",
        strictness_tier="low",
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return result["text"]
