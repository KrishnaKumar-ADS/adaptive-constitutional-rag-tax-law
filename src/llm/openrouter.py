"""Synchronous Groq API client for simple inference calls."""
from openai import OpenAI
from src.config import settings


client = OpenAI(
    api_key=settings.GROQ_API_KEY,
    base_url=settings.GROQ_BASE_URL,
)


def inference_groq(
    prompt: str,
    model: str = None,
):
    """
    Synchronous call to Groq.
    Uses settings.GENERATION_MODEL_FREE by default.
    """
    if model is None:
        model = settings.GENERATION_MODEL_FREE

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0.0,
    )

    return response.choices[0].message.content


# Keep backward-compatible alias
inference_openrouter = inference_groq