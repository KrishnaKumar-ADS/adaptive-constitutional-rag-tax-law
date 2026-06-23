import pytest
from src.evidence.evidence_aggregator import aggregate_evidence, Evidence, EvidenceSet

def test_aggregate_evidence_basic():
    # Mocking Qdrant-like dict output
    results = [
        {
            "metadata": {
                "source": "income_tax_act",
                "section_number": "10",
                "chunk_index": 2,
                "chunk_id": "10_chunk_2"
            },
            "text": "Second part of section 10.",
            "score": 0.8
        },
        {
            "metadata": {
                "source": "income_tax_act",
                "section_number": "10",
                "chunk_index": 1,
                "chunk_id": "10_chunk_1"
            },
            "text": "First part of section 10.",
            "score": 0.9
        }
    ]
    
    evidence_set = aggregate_evidence(results)
    
    assert isinstance(evidence_set, EvidenceSet)
    assert len(evidence_set.evidences) == 1
    
    ev = evidence_set.evidences[0]
    assert ev.citation_id == "10"
    # It should sort by chunk_index
    assert ev.text == "First part of section 10.\nSecond part of section 10."
    # It should take the max score
    assert ev.score == 0.9
    assert ev.source_type == "income_tax_act"
    assert ev.section_number == "10"
