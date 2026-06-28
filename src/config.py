from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "Adaptive Constitutional RAG"
    GROQ_API_KEY: str = ""
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    DATABASE_URL: str = "postgresql://tax_rag:postgres@localhost:5432/tax_rag"
    REDIS_URL: str = "redis://localhost:6379/0"
    QDRANT_URL: str = "http://localhost:6333"

    # Groq model names
    GENERATION_MODEL_FREE: str = "openai/gpt-oss-120b"
    FAST_MODEL_FREE: str = "openai/gpt-oss-20b"
    
    UNCERTAINTY_LOW_THRESHOLD: float = 0.30
    UNCERTAINTY_HIGH_THRESHOLD: float = 0.65

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()