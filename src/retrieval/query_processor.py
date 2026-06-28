"""Query analysis for legal retrieval — section extraction and intent detection."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

Intent = Literal["penalty", "exemption", "deduction", "definition", "procedure", "general"]

SECTION_REF_PATTERN = re.compile(
    r"(?:Section|Sec\.?|Article|Art\.?)\s+(\d+[A-Z]*(?:\(\d+[A-Za-z]*\))?)",
    re.IGNORECASE,
)

PENALTY_KEYWORDS = (
    "penalty", "penalties", "punish", "punishable", "fine", "imprison",
    "prosecution", "offence", "offense", "fee for default", "failure to comply",
    "non-compliance", "non compliance", "default",
)
EXEMPTION_KEYWORDS = ("exempt", "exemption", "not taxable", "not chargeable")
DEDUCTION_KEYWORDS = ("deduction", "deduct", "deductible", "allowance")
DEFINITION_KEYWORDS = ("what is", "what does", "define", "meaning of", "states that")
PROCEDURE_KEYWORDS = ("how to", "procedure", "process", "steps", "file", "furnish", "return")


@dataclass
class ProcessedQuery:
    original: str
    referenced_sections: list[str] = field(default_factory=list)
    referenced_articles: list[str] = field(default_factory=list)
    intent: Intent = "general"
    search_queries: list[str] = field(default_factory=list)


def _normalize_section(raw: str) -> str:
    """Strip subsection suffix: 10(1) -> 10, keep letter suffixes: 206C."""
    return re.sub(r"\(\d+[A-Za-z]*\)", "", raw.strip()).upper()


def detect_intent(query: str) -> Intent:
    q = query.lower()
    if any(k in q for k in PENALTY_KEYWORDS):
        return "penalty"
    if any(k in q for k in EXEMPTION_KEYWORDS):
        return "exemption"
    if any(k in q for k in DEDUCTION_KEYWORDS):
        return "deduction"
    if any(k in q for k in PROCEDURE_KEYWORDS):
        return "procedure"
    if any(k in q for k in DEFINITION_KEYWORDS):
        return "definition"
    return "general"


def process_query(query: str) -> ProcessedQuery:
    """
    Parse a legal question into retrieval-friendly signals.

    - Extracts explicit Section/Article references
    - Detects query intent (penalty, exemption, etc.)
    - Builds expanded search queries for hybrid retrieval
    """
    sections: list[str] = []
    articles: list[str] = []

    for match in SECTION_REF_PATTERN.finditer(query):
        token = match.group(0)
        normalized = _normalize_section(match.group(1))
        if token.lower().startswith(("article", "art")):
            if normalized not in articles:
                articles.append(normalized)
        else:
            if normalized not in sections:
                sections.append(normalized)

    intent = detect_intent(query)
    search_queries = [query]

    if sections:
        sec_label = ", ".join(f"Section {s}" for s in sections)
        if intent == "penalty":
            search_queries.extend([
                f"penalty failure comply {sec_label}",
                f"punishable fine imprisonment section {sections[0]}",
                f"fee default statement section {sections[0]}",
            ])
        elif intent == "exemption":
            search_queries.append(f"exemption not taxable {sec_label}")
        elif intent == "deduction":
            search_queries.append(f"deduction allowance {sec_label}")
        elif intent == "definition":
            search_queries.append(f"{sec_label} provision text")
        else:
            search_queries.append(sec_label)

    if articles:
        art_label = ", ".join(f"Article {a}" for a in articles)
        search_queries.append(art_label)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_queries: list[str] = []
    for q in search_queries:
        key = q.lower().strip()
        if key not in seen:
            seen.add(key)
            unique_queries.append(q)

    return ProcessedQuery(
        original=query,
        referenced_sections=sections,
        referenced_articles=articles,
        intent=intent,
        search_queries=unique_queries,
    )
