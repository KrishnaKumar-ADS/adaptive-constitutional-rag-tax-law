"""Quick verification that all modified modules import and basic logic works."""
import sys
sys.path.insert(0, ".")

# 1. Test inference_groq
from src.llm.inference_groq import extract_allowed_citations, REFORMAT_SYSTEM_PROMPT
test_evidence = "[Section 10] some text about Section 206C and Article 265"
allowed = extract_allowed_citations(test_evidence)
print("OK inference_groq: extract_allowed_citations =", allowed)
assert "Section 10" in allowed, f"Section 10 not found in {allowed}"
assert "Section 206C" in allowed, f"Section 206C not found in {allowed}"
assert "Article 265" in allowed, f"Article 265 not found in {allowed}"
print("  REFORMAT_SYSTEM_PROMPT contains grounding rules:", "CRITICAL GROUNDING RULES" in REFORMAT_SYSTEM_PROMPT)

# 2. Test prompts
from src.llm.prompts import SYSTEM_PROMPT
assert "NEVER invent or fabricate" in SYSTEM_PROMPT, "prompts.py not hardened"
print("OK prompts.py: SYSTEM_PROMPT hardened")

# 3. Test inference_local prompt
from src.llm.inference_local import SYSTEM_PROMPT as LOCAL_PROMPT
assert "NEVER invent, fabricate" in LOCAL_PROMPT, "inference_local.py not hardened"
print("OK inference_local.py: SYSTEM_PROMPT hardened")

# 4. Test constitutional rules
from src.constitutional.rules import check_rules, _check_rule_5_invented_provisions
violations = _check_rule_5_invented_provisions("Section 10 states that...")
print("OK rules.py: Rule 5 check for Section 10 (valid) =", violations)

violations_bad = _check_rule_5_invented_provisions("Section 21F states that...")
print("OK rules.py: Rule 5 check for Section 21F (invalid) =", violations_bad)
assert len(violations_bad) > 0, "Rule 5 should catch Section 21F!"

# 5. Test pipeline import
from src.pipeline import _build_evidence_only_answer
from src.evidence.evidence_aggregator import Evidence, EvidenceSet
test_ev = Evidence(
    citation_id="10", text="Section 10(1) provides exemption for agricultural income.",
    source_type="income_tax_act", score=0.9, section_number="10"
)
fallback = _build_evidence_only_answer("Is agricultural income exempt?", EvidenceSet(evidences=[test_ev]))
print("OK pipeline.py: Evidence-only fallback generated")
print("  First 120 chars:", fallback[:120])

# 6. Test citation verifier import
from src.citation.citation_verifier import CitationVerifier
print("OK citation_verifier.py: Imports OK")

print("")
print("=== ALL CHECKS PASSED ===")
