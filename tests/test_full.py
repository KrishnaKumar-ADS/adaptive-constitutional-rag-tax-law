import pytest
from unittest.mock import patch, MagicMock
from src.pipeline import run_pipeline
from src.evidence.evidence_aggregator import Evidence, EvidenceSet

@patch("src.pipeline.hybrid_search")
@patch("src.pipeline.inference_openrouter")
@patch("src.pipeline.log_query")
def test_full_pipeline(mock_log_query, mock_inference, mock_hybrid_search):
    # Mock retrieval
    mock_point = MagicMock()
    mock_point.payload = {
        "source": "income_tax_act",
        "section_number": "10",
        "chunk_id": "income_tax_section_10_chunk_2",
        "text": "in the case of a non-resident, any income by way of interest on such securities or bonds as the Central Government may specify... is not included in total income."
    }
    mock_point.score = 0.95
    mock_hybrid_search.return_value = [mock_point]
    
    # Mock LLM
    mock_inference.return_value = "Interest income on securities for non-residents is exempt from tax under Section 10."
    
    # Mock DB logging
    mock_log_query.return_value = 1
    
    # Run pipeline
    response = run_pipeline("Is interest income on securities for non-residents taxable?", log_to_db=True)
    
    assert response.question == "Is interest income on securities for non-residents taxable?"
    assert response.answer == "Interest income on securities for non-residents is exempt from tax under Section 10."
    # It should have checked citation and found Valid or Partially Supported depending on NLI entailment 
    # NLI model will actually run in this test, but with mocked text, it should be valid
    assert response.verdict in ["Valid", "Partially Supported"]
    assert response.evidence_count == 1
    assert response.evidence[0].citation_id == "10"
    
    # Verify mocks were called
    mock_hybrid_search.assert_called_once()
    mock_inference.assert_called_once()
    mock_log_query.assert_called_once()
