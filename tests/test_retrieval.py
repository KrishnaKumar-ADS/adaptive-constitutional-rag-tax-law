"""Retrieval evaluation tests using pytest.

Tests hybrid retrieval against hand-written queries with known-correct
expected sections/articles. Cross-reference and query-processor tests
run offline; full hybrid search tests require Qdrant.
"""
import pytest

from src.retrieval.query_processor import process_query
from src.retrieval.cross_reference import find_cross_references


TEST_CASES = [
    {
        "query": "Is agricultural income taxable?",
        "expected_section": "10",
        "expected_article": None,
    },
    {
        "query": "What does Article 265 say about taxation?",
        "expected_section": None,
        "expected_article": "265",
    },
    {
        "query": "What is Permanent Account Number?",
        "expected_section": "139A",
        "expected_article": None,
    },
    {
        "query": "What is the penalty for failure to comply with Section 206C?",
        "expected_section": "276BB",
        "expected_article": None,
    },
    {
        "query": "Income escaping assessment",
        "expected_section": "147",
        "expected_article": None,
    },
    {
        "query": "What is advance tax?",
        "expected_section": "208",
        "expected_article": None,
    },
    {
        "query": "Tax deducted at source salary",
        "expected_section": "192",
        "expected_article": None,
    },
    {
        "query": "Deduction for higher education loan",
        "expected_section": "80E",
        "expected_article": None,
    },
    {
        "query": "Health insurance deduction",
        "expected_section": "80D",
        "expected_article": None,
    },
    {
        "query": "Right to equality",
        "expected_section": None,
        "expected_article": "14",
    },
]


def _check_retrieval(query, expected_section, expected_article, top_k=8):
    """Helper: run hybrid search and check if expected result is in top-k."""
    import src.retrieval.hybrid_retriever as hr

    processed = process_query(query)
    results = hr.hybrid_search(query, top_k=top_k, processed=processed)

    for rank, result in enumerate(results, start=1):
        payload = result.payload

        section = str(payload.get("section_number", "")).strip()
        article = str(payload.get("article_number", "")).strip()

        if expected_section and section == expected_section:
            return True, rank

        if expected_article and article == expected_article:
            return True, rank

    return False, None


@pytest.mark.parametrize(
    "test_case",
    TEST_CASES,
    ids=[tc["query"][:40] for tc in TEST_CASES],
)
def test_query_processor_extracts_references(test_case):
    """Query processor should extract explicit section/article references."""
    processed = process_query(test_case["query"])
    if test_case["expected_section"]:
        assert test_case["expected_section"] in processed.referenced_sections
    if test_case["expected_article"]:
        assert test_case["expected_article"] in processed.referenced_articles


def test_penalty_query_finds_cross_referenced_sections():
    """Penalty questions about 206C should surface cross-referenced penalty sections."""
    processed = process_query(
        "What is the penalty for failure to comply with Section 206C?"
    )
    assert processed.intent == "penalty"
    assert "206C" in processed.referenced_sections

    xrefs = find_cross_references(["206C"], intent="penalty")
    xref_sections = {e["section_number"] for e in xrefs}
    assert "276BB" in xref_sections
    assert "271H" in xref_sections


@pytest.mark.integration
@pytest.mark.parametrize(
    "test_case",
    TEST_CASES,
    ids=[tc["query"][:40] for tc in TEST_CASES],
)
def test_hybrid_retrieval(test_case):
    """Each test case checks that the expected section/article is retrieved."""
    found, rank = _check_retrieval(
        test_case["query"],
        test_case["expected_section"],
        test_case["expected_article"],
    )

    assert found, (
        f"Expected Section={test_case['expected_section']} "
        f"Article={test_case['expected_article']} "
        f"not found in top-8 for query: {test_case['query']}"
    )
