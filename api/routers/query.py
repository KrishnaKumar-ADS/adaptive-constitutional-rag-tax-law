from fastapi import APIRouter
from src.pipeline import run_pipeline_async
from api.schemas import QueryRequest
from src.generation.output_schema import LegalResponse

router = APIRouter()

@router.post("/query", response_model=LegalResponse)
async def query_endpoint(req: QueryRequest):
    """
    Main query endpoint using the full Adaptive Constitutional RAG pipeline.
    """
    response = await run_pipeline_async(
        question=req.question,
        use_local_model=req.use_local_model
    )
    return response
