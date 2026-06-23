"""OpenRouter API client."""
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
    api_key=settings.OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://localhost:3000", 
        "X-Title": "Tax Agent Script",
    }
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

    raise RuntimeError("OpenRouter failed after retries")


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