# src/generation/output_schema.py

from pydantic import BaseModel
from typing import List


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