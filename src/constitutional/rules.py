from src.generation.output_schema import LegalResponse

def check_rules(response: LegalResponse) -> list[str]:
    """
    Checks the generated response against the 5 constitutional rules.
    Returns a list of violation descriptions. Empty list means perfectly compliant.
    
    Rule 1: No unsupported claims (Checked primarily by Layer 7, but we ensure structure here).
    Rule 2: Must cite source.
    Rule 3: Must identify conflicting evidence (if present in prompt).
    Rule 4: Must abstain if insufficient evidence.
    Rule 5: Cannot invent provisions.
    """
    violations = []
    
    # Rule 2: Must cite source if answered
    if response.verdict and response.verdict.startswith("Valid"):
        if not response.evidence:
            violations.append("Rule 2 Violation: Decision is 'Valid' but no citations were provided.")
            
    # Rule 4: Must abstain if insufficient evidence
    if not response.evidence and response.verdict != "Abstained":
        violations.append("Rule 4 Violation: No retrieved evidence, but model did not abstain.")
        
    # Rule 5 is primarily caught by Layer 7 (Citation Verification).
    
    return violations
