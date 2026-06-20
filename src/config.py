"""Central configuration loader using pydantic-settings."""
from pydantic_settings import BaseSettings
class Settings(BaseSettings):
    PROJECT_NAME: str
    OPENROUTER_API_KEY: str = ""
    DATABASE_URL: str = ""
    QDRANT_URL: str = ""
    REDIS_URL: str = ""
    class Config:
        env_file = ".env"
settings = Settings()