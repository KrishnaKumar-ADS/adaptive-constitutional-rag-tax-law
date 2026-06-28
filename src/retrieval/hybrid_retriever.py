"""Hybrid dense+sparse retrieval with RRF fusion and legal query routing."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client import models

from src.retrieval.cross_reference import (
    find_cross_references,
    find_section_chunks,
)
from src.retrieval.embedder import encode_query
from src.retrieval.query_processor import ProcessedQuery, process_query

client = QdrantClient(
    url=os.environ.get("QDRANT_HOST", os.environ.get("QDRANT_URL", "http://localhost:6333"))
)

COLLECTION = "tax_law"

# Score boosts (additive on RRF score, which is typically 0.01–0.05)
BOOST_EXACT_SECTION = 0.15
BOOST_CROSS_REF_PENALTY = 0.12
BOOST_MENTIONS_REFERENCED = 0.06
BOOST_INTENT_KEYWORD = 0.04


@dataclass
class _ScoredPoint:
    point: object
    score: float


def _rrf_search(query: str, limit: int = 20):
    """Run dense+sparse RRF fusion for a single query string."""
    encoded = encode_query(query)
    dense_vector = encoded["dense_vecs"][0]
    sparse = encoded["lexical_weights"][0]

    sparse_vector = models.SparseVector(
        indices=list(sparse.keys()),
        values=list(sparse.values()),
    )

    results = client.query_points(
        collection_name=COLLECTION,
        prefetch=[
            models.Prefetch(
                query=dense_vector,
                using="dense",
                limit=50,
            ),
            models.Prefetch(
                query=sparse_vector,
                using="sparse",
                limit=50,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
    )
    return results.points


def _wrap_point(record, score: float = 0.0):
    """Wrap a Qdrant Record/ScoredPoint into a mutable object with .score."""
    class _Point:
        pass

    pt = _Point()
    pt.id = getattr(record, "id", None)
    pt.payload = getattr(record, "payload", {}) or {}
    pt.score = float(getattr(record, "score", score) or score)
    return pt


def _fetch_by_field(field: str, values: list[str], limit: int = 10):
    """Fetch Qdrant points whose payload field matches any value in the list."""
    if not values:
        return []

    conditions = [
        models.FieldCondition(
            key=field,
            match=models.MatchValue(value=val),
        )
        for val in values
    ]

    results, _ = client.scroll(
        collection_name=COLLECTION,
        scroll_filter=models.Filter(should=conditions),
        limit=limit,
        with_payload=True,
    )
    return [_wrap_point(r, score=0.0) for r in results]


def _chunk_to_point(chunk: dict, score: float):
    """Wrap a chunks.jsonl entry as a pseudo ScoredPoint for aggregation."""
    payload = {
        "chunk_id": chunk.get("chunk_id", ""),
        "text": chunk.get("text", ""),
        **chunk.get("metadata", {}),
    }
    return _wrap_point(type("R", (), {"id": chunk.get("chunk_id"), "payload": payload})(), score)


def _mentions_section(text: str, section: str) -> bool:
    """True if text references the section without matching longer ids (206CA vs 206C)."""
    pattern = rf"(?<![0-9A-Z]){re.escape(section)}(?![0-9A-Z])"
    return bool(re.search(pattern, text, flags=re.IGNORECASE))


def _rescore_point(point, processed: ProcessedQuery) -> float:
    """Apply legal-aware boosts on top of the base RRF score."""
    score = float(getattr(point, "score", 0.0) or 0.0)
    payload = point.payload or {}
    chunk_section = str(payload.get("section_number", "")).upper()
    chunk_article = str(payload.get("article_number", "")).upper()
    chunk_text = payload.get("text", "")
    chunk_title = payload.get("title", "")
    combined = f"{chunk_title} {chunk_text}".lower()

    for sec in processed.referenced_sections:
        if chunk_section == sec:
            score += BOOST_EXACT_SECTION
        elif _mentions_section(combined, sec.lower()):
            score += BOOST_MENTIONS_REFERENCED

    for art in processed.referenced_articles:
        if chunk_article == art:
            score += BOOST_EXACT_SECTION

    if processed.intent == "penalty":
        penalty_terms = ("penalty", "punish", "fine", "imprison", "fee for default")
        if any(t in combined for t in penalty_terms):
            score += BOOST_INTENT_KEYWORD
        if processed.referenced_sections and any(
            _mentions_section(combined, s.lower()) for s in processed.referenced_sections
        ):
            if any(t in combined for t in penalty_terms):
                score += BOOST_CROSS_REF_PENALTY

    return score


def hybrid_search(
    query: str,
    top_k: int = 8,
    processed: ProcessedQuery | None = None,
):
    """
    Hybrid retrieval with query understanding and cross-reference expansion.

    For questions like "penalty for Section 206C", this:
    1. Runs multi-query RRF search with expanded penalty queries
    2. Pulls cross-referenced penalty sections (276BB, 271H, 234E, etc.)
    3. Boosts sections that directly mention the queried section
    """
    if processed is None:
        processed = process_query(query)

    merged: dict[str | int, _ScoredPoint] = {}

    def _add(point, base_score: float | None = None):
        pid = getattr(point, "id", None) or point.payload.get("chunk_id")
        if base_score is not None:
            point.score = base_score
        existing = merged.get(pid)
        if existing is None or float(point.score) > existing.score:
            merged[pid] = _ScoredPoint(point=point, score=float(point.score))

    # 1. Multi-query RRF fusion
    per_query_limit = max(top_k * 3, 20)
    for search_q in processed.search_queries:
        for point in _rrf_search(search_q, limit=per_query_limit):
            _add(_wrap_point(point))

    # 2. Direct fetch of explicitly referenced sections
    if processed.referenced_sections:
        for point in _fetch_by_field("section_number", processed.referenced_sections):
            point.score = merged.get(point.id, _ScoredPoint(point, 0.0)).score + BOOST_EXACT_SECTION
            _add(point, point.score)

        # 3. Cross-reference expansion (critical for penalty/compliance questions)
        xref = find_cross_references(
            processed.referenced_sections,
            intent=processed.intent,
            limit=top_k * 2,
        )
        xref_boost = BOOST_CROSS_REF_PENALTY if processed.intent == "penalty" else BOOST_MENTIONS_REFERENCED
        for entry in xref:
            pt = _chunk_to_point(entry, xref_boost)
            _add(pt, xref_boost)

        # Fallback: load cross-ref chunks directly from jsonl if Qdrant scroll missed them
        if processed.intent == "penalty" and len(xref) < 3:
            for entry in find_section_chunks(
                [e["section_number"] for e in xref],
                limit=top_k,
            ):
                pt = _chunk_to_point(entry, xref_boost)
                _add(pt, xref_boost)

    # 4. Article direct fetch
    if processed.referenced_articles:
        for point in _fetch_by_field("article_number", processed.referenced_articles):
            point.score = merged.get(point.id, _ScoredPoint(point, 0.0)).score + BOOST_EXACT_SECTION
            _add(point, point.score)

    # 5. Legal-aware rescoring
    rescored = []
    for item in merged.values():
        pt = item.point
        pt.score = _rescore_point(pt, processed)
        rescored.append(pt)

    rescored.sort(key=lambda p: float(p.score), reverse=True)
    return rescored[:top_k]
