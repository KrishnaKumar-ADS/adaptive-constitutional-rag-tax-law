import pytest
from src.citation.grounding_checker import GroundingChecker
from src.citation.section_validator import SectionValidator
from src.citation.article_validator import ArticleValidator
from src.evidence.evidence_aggregator import Evidence

def test_grounding_checker_with_evidence_objects():
    ev1 = Evidence(
        citation_id="10",
        text="Agricultural income",
        source_type="income_tax_act",
        score=0.9,
        section_number="10",
        article_number=None
    )
    
    assert GroundingChecker.citation_used("10", [ev1]) == True
    assert GroundingChecker.citation_used("10(1)", [ev1]) == True
    assert GroundingChecker.citation_used("15", [ev1]) == False

def test_grounding_checker_with_dicts():
    ev_dict = {
        "section_number": "10",
        "citation_id": "Section 10",
        "text": "Agricultural income"
    }
    
    assert GroundingChecker.citation_used("10", [ev_dict]) == True
    assert GroundingChecker.citation_used("10(1)", [ev_dict]) == True
    assert GroundingChecker.citation_used("15", [ev_dict]) == False

def test_section_validator_regex():
    sv = SectionValidator(index_path="data/processed/section_index.json") # We assume the file exists since day 3
    # Just checking regex extraction logic without relying on the file heavily
    sections = sv.extract_sections("This is Section 10(1) and Section 14A.")
    assert set(sections) == {"10(1)", "14A"}

def test_article_validator_regex():
    av = ArticleValidator(index_path="data/processed/article_index.json")
    articles = av.extract_articles("Article 265 prohibits tax without law. Article 14 guarantees equality.")
    assert set(articles) == {"265", "14"}
