"""Main pipeline orchestrating all layers of the system.

Day 7 deliverable: End-to-end query → structured response.

Layers used:
  Layer 1: Query processing
  Layer 2: Hybrid retrieval (dense + sparse + RRF)
  Layer 3: Evidence aggregation
  Layer 5: LLM generation (OpenRouter free model)
  Layer 7: Citation verification
  Layer 8: Structured output
"""

from src.retrieval.hybrid_retriever import (
    hybrid_search,
)

from src.evidence.evidence_aggregator import (
    aggregate_evidence,
)

from src.llm.prompts import (
    build_prompt,
)

from src.llm.openrouter import (
    inference_openrouter,
)

from src.citation.citation_verifier import (
    CitationVerifier,
)

from src.generation.response_builder import (
    build_response,
)

from src.db.crud import log_query


# Lazy-loaded verifier singleton
_verifier = None


def _get_verifier():
    global _verifier
    if _verifier is None:
        _verifier = CitationVerifier()
    return _verifier


from src.uncertainty.features import retrieval_confidence, evidence_agreement, coverage, entropy
from src.uncertainty.calibration import load_calibrated_model
from src.constitutional.enforcement_policy import select_strictness
from src.constitutional.rules import check_rules
import numpy as np
from pathlib import Path

# Lazy-loaded uncertainty model
_uncertainty_model = None

def _get_uncertainty_model():
    global _uncertainty_model
    if _uncertainty_model is None:
        model_path = Path("data/processed/calibrated_uncertainty_model.joblib")
        if model_path.exists():
            _uncertainty_model = load_calibrated_model(str(model_path))
        else:
            print("Warning: Calibrated uncertainty model not found. Using dummy 0.0 score.")
            _uncertainty_model = "DUMMY"
    return _uncertainty_model

def run_pipeline(
    question: str,
    top_k: int = 8,
    log_to_db: bool = True,
):
    """
    Full end-to-end pipeline:
      query → hybrid_retriever → evidence_aggregator
      → [NEW] uncertainty computation → enforcement_policy → dynamic prompt
      → inference_openrouter
      → citation_verifier → response_builder
      → [NEW] constitutional rules check
      → (optional) log to Postgres
    """

    # Layer 2: Hybrid retrieval
    retrieved = hybrid_search(question, top_k=top_k)

    # Layer 3: Evidence aggregation
    evidence_set = aggregate_evidence(retrieved)

    # --- Layer 4: Uncertainty Estimation ---
    # Extract features (simulated evidence agreement since we'd normally run cross-encoder on all passages)
    scores = [p.score for p in retrieved]
    
    # 1. Retrieval confidence
    f_conf = retrieval_confidence(scores)
    # 2. Entropy
    f_ent = entropy(scores)
    # 3. Coverage (Naive word overlap for demo)
    query_terms = set(question.lower().split())
    evidence_terms = set()
    for ev in evidence_set.evidences:
        evidence_terms.update(ev.text.lower().split())
    f_cov = coverage(query_terms, evidence_terms)
    # 4. Evidence agreement (Simulated for pipeline demo unless we want to run NLI N^2 times here)
    # For a real pipeline, we'd run NLI between pairs. 
    f_agree = 0.8 # Dummy value for now
    
    X = np.array([[f_conf, f_agree, f_cov, f_ent]])
    
    u_model = _get_uncertainty_model()
    if u_model == "DUMMY":
        u_score = 0.0
    else:
        # Calibrated model predict_proba
        u_score = float(u_model.predict_proba(X)[0][1])
        
    print(f"[Pipeline] Computed Uncertainty (U): {u_score:.3f}")

    # --- Layer 5: Constitutional Engine ---
    strictness = select_strictness(u_score)
    print(f"[Pipeline] Strictness Tier: {strictness.upper()}")
    
    template_path = Path(f"src/constitutional/prompt_templates/{strictness}_uncertainty{'_abstain' if strictness == 'high' else ''}.txt")
    with open(template_path, "r", encoding="utf-8") as f:
        system_template = f.read()
        
    evidence_text = ""
    for idx, ev in enumerate(evidence_set.evidences, start=1):
        evidence_text += f"\nCitation: {ev.citation_id}\n{ev.text}\n"

    # Fill template
    filled_prompt = system_template.replace("{evidence_text}", evidence_text)
    
    # Final prompt for the LLM
    final_prompt = f"{filled_prompt}\n\nQuestion: {question}"

    # Layer 6: LLM generation
    # If high strictness, we can even skip the LLM call entirely to save money, 
    # but the prompt template instructs it to abstain, which handles it nicely.
    answer = inference_openrouter(final_prompt)

    # Layer 7: Citation verification
    verifier = _get_verifier()
    citation_result = verifier.verify(
        claim=answer,
        evidence_list=evidence_set.evidences,
    )

    # Layer 8: Structured response
    response = build_response(
        question=question,
        answer=answer,
        evidence_set=evidence_set,
        citation_result=citation_result,
        model_name="openai/gpt-oss-120b:free",
    )
    
    # Embed the U score into confidence
    response.confidence = round(1.0 - u_score, 3)
    
    # Check Constitutional Rules
    violations = check_rules(response)
    if violations:
        print(f"[Pipeline] Warning: Constitutional Violations Detected: {violations}")

    # Log to Postgres
    if log_to_db:
        try:
            evidence_ids = [ev.citation_id for ev in evidence_set.evidences]
            log_query(
                question=question,
                evidence_ids=evidence_ids,
                decision=response.verdict,
                citations_json=citation_result,
                raw_response=answer,
            )
        except Exception as e:
            print(f"Warning: DB logging failed: {e}")

    return response
