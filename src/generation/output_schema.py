# src/generation/output_schema.py

from pydantic import BaseModel, Field
from typing import List, Optional


class EvidenceReference(BaseModel):
    citation_id: str
    source_type: str
    score: float


class LegalResponse(BaseModel):

    question: str
    answer: str
    verdict: str
    confidence: float
    evidence_count: int
    evidence: List[EvidenceReference]
    retrieval_method: str
    model_name: str
    
    # Fields added for baseline evaluation (Days 13-14)
    decision: Optional[str] = None
    uncertainty_score: Optional[float] = None
    strictness_tier: Optional[str] = None
    processing_time_ms: Optional[float] = None
    citations_verified: Optional[List[dict]] = None

    # Fields added for Groq reformatting (post-processing)
    quality_score: Optional[float] = None
    issues_found: Optional[List[str]] = None
    reformatted: Optional[bool] = None