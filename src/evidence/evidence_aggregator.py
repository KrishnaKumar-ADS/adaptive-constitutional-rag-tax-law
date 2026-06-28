"""Aggregate and deduplicate retrieved evidence."""
# src/evidence/evidence_aggregator.py

from dataclasses import dataclass
from typing import List
from collections import defaultdict


@dataclass
class Evidence:
    citation_id: str
    text: str
    source_type: str
    score: float
    section_number: str = None
    article_number: str = None


@dataclass
class EvidenceSet:
    evidences: List[Evidence]


def _extract_from_result(result):
    """
    Extract a normalised dict from either a Qdrant ScoredPoint
    or a plain dict.  This bridges the format gap between
    hybrid_search() output and what the rest of the pipeline expects.
    """
    # Qdrant ScoredPoint objects
    if hasattr(result, "payload"):
        payload = result.payload
        return {
            "metadata": {
                k: v for k, v in payload.items()
                if k != "text"
            },
            "text": payload.get("text", ""),
            "score": getattr(result, "score", 0.0),
        }

    # Already a dict
    return result


def aggregate_evidence(results) -> EvidenceSet:
    """
    Takes hybrid retrieval's Top-K results (ScoredPoint or dict),
    deduplicates overlapping chunks from the same section/article,
    groups by source type, and returns a clean EvidenceSet.
    """
    normalised = [_extract_from_result(r) for r in results]

    grouped = defaultdict(list)

    for r in normalised:
        key = (
            r["metadata"].get("section_number")
            or r["metadata"].get("article_number")
            or r["metadata"].get("chunk_id", "unknown")
        )
        grouped[key].append(r)

    evidences = []

    for citation_id, chunks in grouped.items():

        chunks = sorted(
            chunks,
            key=lambda x: x["metadata"].get("chunk_index", 0),
        )

        merged_text = "\n".join(
            c["text"] for c in chunks
        )

        best_score = max(
            c["score"] for c in chunks
        )

        source_type = chunks[0]["metadata"].get("source", "unknown")
        section_number = chunks[0]["metadata"].get("section_number")
        article_number = chunks[0]["metadata"].get("article_number")

        if section_number:
            display_id = f"Section {section_number}"
        elif article_number:
            display_id = f"Article {article_number}"
        else:
            display_id = str(citation_id)

        evidences.append(
            Evidence(
                citation_id=display_id,
                text=merged_text,
                source_type=source_type,
                score=best_score,
                section_number=section_number,
                article_number=article_number,
            )
        )

    evidences.sort(
        key=lambda x: x.score,
        reverse=True,
    )

    return EvidenceSet(evidences=evidences)