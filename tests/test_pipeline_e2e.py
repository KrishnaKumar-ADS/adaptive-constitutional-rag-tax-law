import pytest
import asyncio
from src.pipeline import run_pipeline_async
from src.generation.output_schema import LegalResponse

@pytest.mark.asyncio
async def test_end_to_end_pipeline():
    """
    Test the full async pipeline end-to-end.
    Requires Groq API key and services to be running (or mocked).
    """
    question = "What is the penalty for late filing under section 271F?"
    
    # Run the pipeline
    response = await run_pipeline_async(question, use_local_model=False)
    
    # Assertions
    assert isinstance(response, LegalResponse)
    assert response.question == question
    assert response.verdict in ["Valid", "Invalid", "Partially Supported", "Abstained"]
    assert hasattr(response, "decision")
    assert hasattr(response, "uncertainty_score")
    assert hasattr(response, "strictness_tier")
    
    # New Groq reformatter fields
    assert hasattr(response, "quality_score")
    assert hasattr(response, "issues_found")
    assert hasattr(response, "reformatted")
    
    # We should have evidence returned
    if response.decision != "Abstained":
        assert response.evidence_count > 0
        assert len(response.evidence) > 0
