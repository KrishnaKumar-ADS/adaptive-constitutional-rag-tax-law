from fastapi import APIRouter
from api.schemas import HealthResponse
from src.llm.inference_local import check_ollama_health

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check endpoint."""
    # In a full system you would check Redis/Qdrant/PG here
    ollama_status = check_ollama_health()
    
    return HealthResponse(
        status="healthy" if ollama_status["healthy"] else "degraded",
        ollama_healthy=ollama_status["healthy"],
        qdrant_healthy=True, # Mocked for now
        redis_healthy=True,
        postgres_healthy=True
    )
