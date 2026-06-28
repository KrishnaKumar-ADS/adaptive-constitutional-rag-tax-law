"""
Groq API client — replaces the OpenRouter client.

Provides:
  - Async generation with rate limiting + exponential backoff
  - generate_groq()        — standard generation with evidence injection
  - generate_groq_raw()    — raw generation without evidence (Baseline A)
  - reformat_and_score()   — takes raw Qwen3-8B output and reformats it
                             via Groq gpt-oss-120b, also scoring quality
"""

import asyncio
import json
import re
import time

from openai import AsyncOpenAI

from src.config import settings


def _log(msg: str):
    """Print safely on Windows cp1252."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"))

# --- System prompt shared across generation calls ---
try:
    from src.llm.prompts import SYSTEM_PROMPT
except ImportError:
    SYSTEM_PROMPT = (
        "You are a helpful AI tax assistant. "
        "Answer the question based ONLY on the provided evidence."
    )

# --- Groq async client (OpenAI-compatible) ---
client = AsyncOpenAI(
    api_key=settings.GROQ_API_KEY,
    base_url=settings.GROQ_BASE_URL,
)

REQUEST_LIMITER = asyncio.Semaphore(20)


# ── Core LLM caller with retries ────────────────────────────────────────────

async def _call_llm(model: str, messages: list, temperature: float = 0.1, max_tokens: int = 1024) -> str:
    """Low-level call with exponential backoff on failures."""
    retries = 5

    for attempt in range(retries):
        try:
            async with REQUEST_LIMITER:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=60.0,
                )
                return response.choices[0].message.content

        except Exception as e:
            _log(f"[Groq] Attempt {attempt + 1} failed: {e}")
            wait = 2 ** attempt
            await asyncio.sleep(wait)

    raise RuntimeError("Groq API failed after retries")


# ── Standard generation (replaces generate_openrouter) ──────────────────────

async def generate_groq(
    question: str,
    evidence_text: str = "",
    strictness_tier: str = "medium",
    temperature: float = 0.1,
    max_tokens: int = 1024,
) -> dict:
    """
    Async Groq call with evidence injection, matching the
    interface of inference_local.generate_local().
    """
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
    text = await _call_llm(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
    elapsed_ms = (time.perf_counter() - start) * 1000

    return {
        "text": text,
        "model": model,
        "wall_time_ms": elapsed_ms,
    }


# ── Raw generation (Baseline A) ────────────────────────────────────────────

async def generate_groq_raw(
    question: str,
    temperature: float = 0.1,
    max_tokens: int = 1024,
) -> str:
    """
    Raw Groq generation without evidence — used by Baseline A.
    Returns just the text string.
    """
    result = await generate_groq(
        question=question,
        evidence_text="",
        strictness_tier="low",
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return result["text"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def extract_allowed_citations(evidence_text: str) -> list[str]:
    """
    Parse the evidence text to extract every Section/Article number present.
    Returns a deduplicated, sorted list like ['Section 10', 'Section 206C', 'Article 265'].
    """
    section_pattern = re.compile(r"Section\s+(\d+[A-Z]*(?:\(\d+\))?)", re.IGNORECASE)
    article_pattern = re.compile(r"Article\s+(\d+[A-Z]*(?:\(\d+\))?)", re.IGNORECASE)

    sections = {f"Section {m}" for m in section_pattern.findall(evidence_text)}
    articles = {f"Article {m}" for m in article_pattern.findall(evidence_text)}

    # Also extract from [Section X] tags used in flat_evidence formatting
    bracket_pattern = re.compile(r"\[Section\s+(\d+[A-Z]*(?:\(\d+\))?)\]", re.IGNORECASE)
    sections.update(f"Section {m}" for m in bracket_pattern.findall(evidence_text))

    return sorted(sections | articles)


# ── Reformat + Score (post-processes Qwen3-8B raw output) ────────────────────

REFORMAT_SYSTEM_PROMPT = """\
You are a legal editor AI. Your ONLY job is to clean up the formatting of a \
raw AI-generated answer about Indian tax law.

## CRITICAL GROUNDING RULES — VIOLATION IS UNACCEPTABLE

1. You may ONLY reference Section or Article numbers that appear in the \
   "ALLOWED CITATIONS" list below. If a citation is NOT in that list, you MUST \
   remove it from the answer and record it in "hallucinated_citations".
2. You must NOT add any legal provisions, exceptions, caveats, procedural \
   details, or commentary from your own knowledge — ONLY from the evidence.
3. You must NOT invent, fabricate, or guess any Section or Article numbers.
4. If you are unsure whether a statement is supported by the evidence, \
   REMOVE it rather than keep it.
5. If removing hallucinated content leaves no substantive answer, set \
   "reformatted_answer" to "I cannot answer this based on available provisions." \
   and "quality_score" to 0.0.

## YOUR TASKS

1. **Reformat** the answer into a clean structure:
   - Direct answer at the top
   - Legal provisions cited in structured format (ONLY from allowed list)
   - Brief explanation derived ONLY from the evidence text
   - Caveats ONLY if stated in the evidence

2. **Score** the quality of the ORIGINAL raw answer (0.0 to 1.0):
   - 1.0 = Accurate, well-cited from evidence, comprehensive
   - 0.7-0.9 = Good with minor formatting issues
   - 0.4-0.6 = Partially correct or missing citations from evidence
   - 0.0-0.3 = Contains hallucinated sections, incorrect statements, or \
     material not in evidence

3. **Identify issues** in the original answer.

4. **List hallucinated citations** — any Section/Article in the raw answer \
   that is NOT in the allowed citations list.

## RESPONSE FORMAT

Return ONLY valid JSON (no markdown fencing):
{
  "reformatted_answer": "The cleaned answer using ONLY evidence-grounded content...",
  "quality_score": 0.85,
  "issues": ["issue 1", "issue 2"],
  "hallucinated_citations": ["Section 21F", "Section 274"]
}

If the original answer is an abstention ("I cannot answer..."), preserve it, \
score 1.0, empty issues and hallucinated_citations.\
"""


async def reformat_and_score(
    raw_answer: str,
    question: str,
    evidence_text: str = "",
) -> dict:
    """
    Send the raw Qwen3-8B output to Groq for evidence-grounded
    reformatting and quality scoring.

    The reformatter is given an explicit whitelist of allowed citations
    extracted from the evidence, and must strip anything not in that list.

    Returns:
        dict with keys:
          - reformatted_answer (str): The cleaned-up, grounded answer
          - quality_score (float): 0.0 – 1.0 quality rating
          - issues (list[str]): Any issues found in the original
          - hallucinated_citations (list[str]): Citations stripped by reformatter
    """
    # Build the allowed-citations whitelist from evidence
    allowed = extract_allowed_citations(evidence_text)
    allowed_str = ", ".join(allowed) if allowed else "(none — no legal provisions in evidence)"

    user_content = (
        f"## ALLOWED CITATIONS (from retrieved evidence)\n{allowed_str}\n\n"
        f"## Original Question\n{question}\n\n"
    )
    if evidence_text:
        user_content += f"## Evidence Provided to the Model\n{evidence_text}\n\n"
    user_content += f"## Raw Model Answer (to reformat and score)\n{raw_answer}"

    messages = [
        {"role": "system", "content": REFORMAT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    model = settings.GENERATION_MODEL_FREE

    try:
        raw_json = await _call_llm(
            model=model,
            messages=messages,
            temperature=0.0,   # Deterministic for formatting
            max_tokens=2048,
        )

        # Strip markdown fencing if model wraps it
        cleaned = raw_json.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        result = json.loads(cleaned)

        hallucinated = result.get("hallucinated_citations", [])
        if hallucinated:
            _log(f"[Groq Reformat] Stripped hallucinated citations: {hallucinated}")

        return {
            "reformatted_answer": result.get("reformatted_answer", raw_answer),
            "quality_score": float(result.get("quality_score", 0.5)),
            "issues": result.get("issues", []),
            "hallucinated_citations": hallucinated,
        }

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        _log(f"[Groq Reformat] JSON parse error: {e}. Returning raw answer.")
        return {
            "reformatted_answer": raw_answer,
            "quality_score": 0.5,
            "issues": [f"Reformatter failed to produce valid JSON: {e}"],
            "hallucinated_citations": [],
        }
    except Exception as e:
        _log(f"[Groq Reformat] API error: {e}. Returning raw answer.")
        return {
            "reformatted_answer": raw_answer,
            "quality_score": 0.5,
            "issues": [f"Reformatter API call failed: {e}"],
            "hallucinated_citations": [],
        }
