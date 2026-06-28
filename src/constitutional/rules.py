"""Constitutional rules enforcement.

Checks the generated response against the 5 constitutional rules.
Returns a list of violation descriptions. Empty list means perfectly compliant.

Rule 1: No unsupported claims (Checked primarily by Layer 7).
Rule 2: Must cite source.
Rule 3: Must identify conflicting evidence (if present in prompt).
Rule 4: Must abstain if insufficient evidence.
Rule 5: Cannot invent provisions — cross-checks against section_index.json.
"""

import json
import re
from pathlib import Path
from src.generation.output_schema import LegalResponse

# Lazy-loaded section/article indexes for Rule 5
_section_index = None
_article_index = None


def _load_section_index():
    global _section_index
    if _section_index is None:
        path = Path("data/processed/section_index.json")
        if path.exists():
            _section_index = json.loads(path.read_text(encoding="utf-8"))
        else:
            _section_index = {}
            print("Warning: section_index.json not found for Rule 5 enforcement")
    return _section_index


def _load_article_index():
    global _article_index
    if _article_index is None:
        path = Path("data/processed/article_index.json")
        if path.exists():
            _article_index = json.loads(path.read_text(encoding="utf-8"))
        else:
            _article_index = {}
            print("Warning: article_index.json not found for Rule 5 enforcement")
    return _article_index


SECTION_PATTERN = re.compile(r"Section\s+(\d+[A-Z]*(?:\(\d+\))?)", re.IGNORECASE)
ARTICLE_PATTERN = re.compile(r"Article\s+(\d+[A-Z]*(?:\(\d+\))?)", re.IGNORECASE)


def _check_rule_5_invented_provisions(answer_text: str) -> list[str]:
    """
    Rule 5: Cannot invent provisions.
    
    Cross-checks every Section/Article number mentioned in the answer
    against the ground-truth section_index.json and article_index.json.
    """
    violations = []
    section_index = _load_section_index()
    article_index = _load_article_index()
    
    # Check sections
    sections_cited = set(SECTION_PATTERN.findall(answer_text))
    for sec in sections_cited:
        # Try exact match and base section (strip sub-section)
        base_sec = re.sub(r"\(\d+\)", "", sec)
        key_exact = f"Section {sec}"
        key_base = f"Section {base_sec}"
        
        if key_exact not in section_index and key_base not in section_index:
            violations.append(
                f"Rule 5 Violation: Section {sec} cited in answer does not exist "
                f"in the Income Tax Act ground-truth index."
            )
    
    # Check articles
    articles_cited = set(ARTICLE_PATTERN.findall(answer_text))
    for art in articles_cited:
        base_art = re.sub(r"\(\d+\)", "", art)
        key_exact = f"Article {art}"
        key_base = f"Article {base_art}"
        
        if key_exact not in article_index and key_base not in article_index:
            violations.append(
                f"Rule 5 Violation: Article {art} cited in answer does not exist "
                f"in the Constitution ground-truth index."
            )
    
    return violations


def check_rules(response: LegalResponse) -> list[str]:
    """
    Checks the generated response against the 5 constitutional rules.
    Returns a list of violation descriptions. Empty list means perfectly compliant.
    """
    violations = []
    
    # Rule 2: Must cite source if answered
    if response.verdict and response.verdict.startswith("Valid"):
        if not response.evidence:
            violations.append(
                "Rule 2 Violation: Decision is 'Valid' but no citations were provided."
            )
            
    # Rule 4: Must abstain if insufficient evidence
    if not response.evidence and response.verdict != "Abstained":
        violations.append(
            "Rule 4 Violation: No retrieved evidence, but model did not abstain."
        )
    
    # Rule 5: Cannot invent provisions (active enforcement)
    if response.answer and response.verdict != "Abstained":
        rule5_violations = _check_rule_5_invented_provisions(response.answer)
        violations.extend(rule5_violations)
        
    return violations
