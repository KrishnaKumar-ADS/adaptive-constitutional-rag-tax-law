"""Tests for citation whitelist enforcement."""
from src.generation.citation_enforcer import enforce_allowed_citations


def test_strips_unauthorized_sections():
    answer = (
        "Under Section 276BB, imprisonment applies. "
        "Also see Section 999Z for extra penalties."
    )
    allowed = ["Section 276BB"]
    cleaned, stripped = enforce_allowed_citations(answer, allowed)
    assert "999Z" not in cleaned
    assert "Section 999Z" in stripped
    assert "276BB" in cleaned


def test_abstains_when_all_blocks_removed():
    answer = "Section 271H and Section 234E apply."
    allowed = ["Section 206CA"]
    cleaned, stripped = enforce_allowed_citations(answer, allowed)
    assert "cannot answer" in cleaned.lower()
    assert len(stripped) == 2
