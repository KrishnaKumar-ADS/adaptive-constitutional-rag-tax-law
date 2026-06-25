from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "Adaptive Constitutional RAG"
    OPENROUTER_API_KEY: str = ""
    DATABASE_URL: str = "postgresql://tax_rag:postgres@localhost:5432/tax_rag"
    REDIS_URL: str = "redis://localhost:6379/0"
    QDRANT_URL: str = "http://localhost:6333"

    # OpenRouter model names (guide2.md spec)
    GENERATION_MODEL_FREE: str = "openai/gpt-oss-120b:free"
    FAST_MODEL_FREE: str = "openai/gpt-oss-20b:free"
    
    UNCERTAINTY_LOW_THRESHOLD: float = 0.30
    UNCERTAINTY_HIGH_THRESHOLD: float = 0.65

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()