from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class QueryRequest(BaseModel):
    question: str
    use_local_model: bool = True
    session_id: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    ollama_healthy: bool
    qdrant_healthy: bool
    redis_healthy: bool
    postgres_healthy: bool
    version: str = "1.0.0"

class EvaluationRequest(BaseModel):
    baseline_id: str = "E"
    question: str
