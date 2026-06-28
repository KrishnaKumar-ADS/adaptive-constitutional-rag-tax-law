"""Groq API client (async).

Backward-compatible module — previously inference_openrouter.py.
All functions now route through Groq (https://api.groq.com/openai/v1).
"""
import asyncio
from openai import AsyncOpenAI

from src.config import settings

# --- IMPORTANT --- 
# If you import your SYSTEM_PROMPT from elsewhere, do it here. 
# Otherwise, keep this fallback so the script doesn't crash.
try:
    from src.llm.prompts import SYSTEM_PROMPT
except ImportError:
    SYSTEM_PROMPT = "You are a helpful AI tax assistant. Answer the question based ONLY on the provided evidence."

client = AsyncOpenAI(
    api_key=settings.GROQ_API_KEY,
    base_url=settings.GROQ_BASE_URL,
)

REQUEST_LIMITER = asyncio.Semaphore(20)

async def _call_llm(model, messages):
    retries = 5

    for attempt in range(retries):
        try:
            async with REQUEST_LIMITER:
                # Timeout added to prevent infinite hanging
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    timeout=60.0 
                )

                return response.choices[0].message.content

        except Exception as e:
            # Print the error so it doesn't fail silently
            print(f"Attempt {attempt + 1} failed: {e}")
            
            wait = 2 ** attempt
            await asyncio.sleep(wait)

    raise RuntimeError("Groq API failed after retries")


async def generate_answer(question, prompt, fast=False):
    """Generates an answer using the requested LLM."""
    
    # Selects the model based on your settings
    model = (
        settings.FAST_MODEL_FREE
        if fast
        else settings.GENERATION_MODEL_FREE
    )

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    # Passes the payload up to your robust caller
    return await _call_llm(
        model=model,
        messages=messages,
    )


# ── Unified wrappers (used by base_model_loader.py and baselines) ──────────

async def generate_openrouter(
    question: str,
    evidence_text: str = "",
    strictness_tier: str = "medium",
    temperature: float = 0.1,
    max_tokens: int = 1024,
) -> dict:
    """
    Async Groq call with evidence injection, matching the
    interface of inference_local.generate_local().
    
    Note: Function name kept as generate_openrouter for backward compatibility
    with baselines. Actually routes through Groq.
    """
    import time

    # Build prompt with evidence
    if evidence_text:
        user_content = (
            f"Context (retrieved legal provisions):\n{evidence_text}\n\n"
            f"Question: {question}"
        )
    else:
        user_content = question

    # Tier-specific system prompt suffix
    tier_suffix = {
        "low": " Provide a detailed answer with full reasoning.",
        "medium": " Flag any conflicts or ambiguity in the evidence.",
        "high": (
            " INSUFFICIENT EVIDENCE. You MUST abstain. Respond only with: "
            "'I cannot answer this based on available provisions.'"
        ),
    }

    system = SYSTEM_PROMPT + tier_suffix.get(strictness_tier, "")

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]

    model = settings.GENERATION_MODEL_FREE

    start = time.perf_counter()
    text = await _call_llm(model=model, messages=messages)
    elapsed_ms = (time.perf_counter() - start) * 1000

    return {
        "text": text,
        "model": model,
        "wall_time_ms": elapsed_ms,
    }


# Alias for new code
generate_groq = generate_openrouter


async def generate_openrouter_raw(
    question: str,
    temperature: float = 0.1,
    max_tokens: int = 1024,
) -> str:
    """
    Raw Groq generation without evidence — used by Baseline A.
    Returns just the text string.
    """
    result = await generate_openrouter(
        question=question,
        evidence_text="",
        strictness_tier="low",
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return result["text"]


# Alias for new code
generate_groq_raw = generate_openrouter_raw