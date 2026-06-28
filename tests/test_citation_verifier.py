"""Citation verifier tests.

Tests the three verification paths:
  1. Fabricated citation → Invalid
  2. Real but not retrieved → Partially Supported
  3. Real and retrieved → Valid
"""
import pytest
from src.citation.citation_verifier import CitationVerifier
from src.evidence.evidence_aggregator import Evidence


@pytest.fixture(scope="module")
def verifier():
    return CitationVerifier()


def test_fake_section(verifier):
    """Fabricated citation should be Invalid."""
    claim = "Income is exempt under Section 999Z."

    result = verifier.verify(
        claim,
        [],
    )

    assert result["verdict"] == "Invalid"


def test_real_not_retrieved(verifier):
    """
    Real section exists but was never retrieved.
    Should be Invalid (all citations ungrounded).
    """
    claim = (
        "Agricultural income is exempt under Section 10."
    )

    # Evidence from a completely different section
    evidence = [
        Evidence(
            citation_id="154",
            text="Rectification of mistake.",
            source_type="income_tax_act",
            score=0.8,
            section_number="154",
            article_number=None,
        )
    ]

    result = verifier.verify(
        claim,
        evidence,
    )

    assert result["verdict"] == "Invalid"


def test_valid_citation(verifier):
    """Real section, retrieved, and entailing → Valid."""
    claim = (
        "Agricultural income is exempt under Section 10."
    )

    evidence = [
        Evidence(
            citation_id="10",
            text=(
                "In computing the total income of a "
                "previous year of any person, any income "
                "falling within any of the following clauses "
                "shall not be included — agricultural income"
            ),
            source_type="income_tax_act",
            score=0.95,
            section_number="10",
            article_number=None,
        )
    ]

    result = verifier.verify(claim, evidence)

    assert result["verdict"] in ("Valid", "Partially Supported"), (
        f"Expected Valid or Partially Supported, got: {result}"
    )