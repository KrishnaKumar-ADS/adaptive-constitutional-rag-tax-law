from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str
    OPENROUTER_API_KEY: str = ""
    DATABASE_URL: str
    REDIS_URL: str
    QDRANT_URL: str
    class Config:
        env_file = ".env"
        extra="ignore"

settings = Settings()