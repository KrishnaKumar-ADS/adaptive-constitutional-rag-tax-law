"""Cross-reference index: find sections that cite a given section/article."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

CHUNKS_FILE = Path("data/processed/chunks.jsonl")

SECTION_MENTION_PATTERN = re.compile(
    r"section\s+(\d+[A-Z]*)",
    re.IGNORECASE,
)

PENALTY_SIGNALS = re.compile(
    r"penalty|punish|fine|imprison|prosecution|offence|offense|fee for default",
    re.IGNORECASE,
)


@lru_cache(maxsize=1)
def _load_cross_reference_index() -> dict[str, list[dict]]:
    """
    Build a reverse index: referenced_section -> citing sections.

    Each entry: {section_number, title, chunk_id, text, is_penalty}
    """
    index: dict[str, dict[str, dict]] = {}

    if not CHUNKS_FILE.exists():
        return {}

    with CHUNKS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            md = chunk.get("metadata", {})
            citing_section = str(md.get("section_number", "")).strip().upper()
            if not citing_section:
                continue

            text = chunk.get("text", "")
            title = md.get("title", "")
            combined = f"{title} {text}"

            for match in SECTION_MENTION_PATTERN.finditer(combined):
                ref = match.group(1).upper()
                if ref == citing_section:
                    continue

                entry = {
                    "section_number": citing_section,
                    "title": title,
                    "chunk_id": chunk.get("chunk_id", ""),
                    "text": text,
                    "is_penalty": bool(PENALTY_SIGNALS.search(combined)),
                }

                bucket = index.setdefault(ref, {})
                # Keep highest-signal chunk per citing section
                existing = bucket.get(citing_section)
                if existing is None or (
                    entry["is_penalty"] and not existing["is_penalty"]
                ):
                    bucket[citing_section] = entry

    return {ref: list(entries.values()) for ref, entries in index.items()}


def find_cross_references(
    section_numbers: list[str],
    intent: str = "general",
    limit: int = 12,
) -> list[dict]:
    """
    Return sections that reference the given section numbers.

    For penalty intent, prioritizes sections with penalty-related language.
    """
    index = _load_cross_reference_index()
    results: list[dict] = []
    seen: set[str] = set()

    for sec in section_numbers:
        sec = sec.upper()
        for entry in index.get(sec, []):
            citing = entry["section_number"]
            if citing in seen:
                continue
            if intent == "penalty" and not entry["is_penalty"]:
                continue
            seen.add(citing)
            results.append(entry)

    # If penalty filter was too strict, fall back to all cross-refs
    if not results and intent == "penalty":
        for sec in section_numbers:
            for entry in index.get(sec.upper(), []):
                citing = entry["section_number"]
                if citing not in seen:
                    seen.add(citing)
                    results.append(entry)

    return results[:limit]


def find_section_chunks(section_numbers: list[str], limit: int = 6) -> list[dict]:
    """Direct lookup of chunks by section_number from chunks.jsonl."""
    if not CHUNKS_FILE.exists() or not section_numbers:
        return []

    targets = {s.upper() for s in section_numbers}
    results: list[dict] = []

    with CHUNKS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            md = chunk.get("metadata", {})
            sec = str(md.get("section_number", "")).strip().upper()
            if sec in targets:
                results.append({
                    "section_number": sec,
                    "title": md.get("title", ""),
                    "chunk_id": chunk.get("chunk_id", ""),
                    "text": chunk.get("text", ""),
                    "metadata": md,
                    "is_penalty": bool(
                        PENALTY_SIGNALS.search(
                            f"{md.get('title', '')} {chunk.get('text', '')}"
                        )
                    ),
                })

    return results[:limit]
