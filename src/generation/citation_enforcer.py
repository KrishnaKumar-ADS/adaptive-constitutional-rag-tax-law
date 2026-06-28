"""Hard enforcement of citation whitelist on generated answers."""
from __future__ import annotations

import re

SECTION_CITATION = re.compile(
    r"\b(?:Section|Sec\.?)\s+(\d+[A-Z]*(?:\(\d+[A-Za-z]*\))?)\b",
    re.IGNORECASE,
)
ARTICLE_CITATION = re.compile(
    r"\b(?:Article|Art\.?)\s+(\d+[A-Za-z]*(?:\(\d+[A-Za-z]*\))?)\b",
    re.IGNORECASE,
)


def _normalize_citation(kind: str, number: str) -> str:
    number = re.sub(r"\(\d+[A-Za-z]*\)", "", number.strip()).upper()
    return f"{kind} {number}"


def extract_citations(text: str) -> list[str]:
    citations = []
    for m in SECTION_CITATION.finditer(text):
        citations.append(_normalize_citation("Section", m.group(1)))
    for m in ARTICLE_CITATION.finditer(text):
        citations.append(_normalize_citation("Article", m.group(1)))
    return citations


def _is_allowed(citation: str, allowed_set: set[str], allowed_base: set[str]) -> bool:
    if citation in allowed_set:
        return True
    base = re.sub(r"\(\d+[A-Za-z]*\)", "", citation.split(" ", 1)[-1]).upper()
    kind = citation.split(" ", 1)[0]
    for a in allowed_set:
        if a.startswith(kind) and base in a.upper():
            return True
    return base in allowed_base


def enforce_allowed_citations(
    answer: str,
    allowed: list[str],
    abstention: str = "I cannot answer this based on available provisions.",
) -> tuple[str, list[str]]:
    """
    Remove citations not in the allowed list from the answer.

    Returns (cleaned_answer, stripped_citations).
    If nothing substantive remains, returns abstention.
    """
    if not answer.strip():
        return abstention, []

    allowed_set = {a.strip() for a in allowed}
    allowed_base = {
        re.sub(r"\(\d+[A-Za-z]*\)", "", a.replace("Section ", "").replace("Article ", "")).upper()
        for a in allowed_set
    }
    stripped: list[str] = []
    kept: list[str] = []

    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    for sentence in sentences:
        original_cites = extract_citations(sentence)
        s = SECTION_CITATION.sub(
            lambda m: m.group(0)
            if _is_allowed(_normalize_citation("Section", m.group(1)), allowed_set, allowed_base)
            else (stripped.append(_normalize_citation("Section", m.group(1))) or ""),
            sentence,
        )
        s = ARTICLE_CITATION.sub(
            lambda m: m.group(0)
            if _is_allowed(_normalize_citation("Article", m.group(1)), allowed_set, allowed_base)
            else (stripped.append(_normalize_citation("Article", m.group(1))) or ""),
            s,
        )
        s = re.sub(r"\s{2,}", " ", s).strip(" ,;—-")

        if not s or len(s) < 15:
            continue
        if original_cites and not extract_citations(s):
            continue
        kept.append(s)

    cleaned = " ".join(kept).strip()

    if len(cleaned) < 20:
        return abstention, sorted(set(stripped))

    return cleaned, sorted(set(stripped))
