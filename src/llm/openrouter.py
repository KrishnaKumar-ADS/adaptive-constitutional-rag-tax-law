"""Synchronous OpenRouter API client for simple inference calls."""
from openai import OpenAI
from src.config import settings


client = OpenAI(
    api_key=settings.OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://localhost:3000",
        "X-Title": "Tax Agent Script",
    },
)


def inference_openrouter(
    prompt: str,
    model: str = None,
):
    """
    Synchronous call to OpenRouter.
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