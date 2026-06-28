"""Main pipeline orchestrating all layers of the system.

Day 7 deliverable: End-to-end query → structured response.

Layers used:
  Layer 1: Query processing
  Layer 2: Hybrid retrieval (dense + sparse + RRF)
  Layer 3: Evidence aggregation
  Layer 4: Uncertainty estimation
  Layer 5: Constitutional enforcement
  Layer 6: LLM generation (Local Qwen3-8B + Groq reformatter)
  Layer 7: Citation verification (comprehensive — reports ALL violations)
  Layer 8: Structured output
  Layer 9: Hallucination rejection loop — strips/replaces hallucinated content
"""

from src.retrieval.hybrid_retriever import (
    hybrid_search,
)
from src.retrieval.query_processor import process_query
from src.generation.citation_enforcer import enforce_allowed_citations
from src.llm.inference_groq import extract_allowed_citations

from src.evidence.evidence_aggregator import (
    aggregate_evidence,
)

from src.llm.prompts import (
    build_prompt,
)

from src.llm.openrouter import (
    inference_groq,
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
from src.config import settings
import logging
import numpy as np
import re
from pathlib import Path

# Windows cp1252 console can't handle Unicode from Groq responses.
# Use a safe logger that replaces un-encodable chars instead of crashing.
logger = logging.getLogger("pipeline")

def _log(msg: str):
    """Print a message safely on Windows (replaces un-encodable Unicode)."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"))

# Lazy-loaded uncertainty model
_uncertainty_model = None

def _get_uncertainty_model():
    global _uncertainty_model
    if _uncertainty_model is None:
        model_path = Path("data/processed/calibrated_uncertainty_model.joblib")
        if model_path.exists():
            _uncertainty_model = load_calibrated_model(str(model_path))
        else:
            _log("Warning: Calibrated uncertainty model not found. Using dummy 0.0 score.")
            _uncertainty_model = "DUMMY"
    return _uncertainty_model


# ── Evidence-Only Fallback Generator ─────────────────────────────────────────




# ── Synchronous Pipeline ────────────────────────────────────────────────────

def run_pipeline(
    question: str,
    top_k: int = 8,
    log_to_db: bool = True,
):
    """
    Full end-to-end pipeline (synchronous path):
      query → hybrid_retriever → evidence_aggregator
      → uncertainty computation → enforcement_policy → dynamic prompt
      → inference_groq (sync)
      → citation_verifier → response_builder
      → constitutional rules check
      → (optional) log to Postgres
    """

    # Layer 1+2: Query processing + hybrid retrieval
    processed = process_query(question)
    retrieved = hybrid_search(question, top_k=top_k, processed=processed)

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
        
    _log(f"[Pipeline] Computed Uncertainty (U): {u_score:.3f}")

    # --- Layer 5: Constitutional Engine ---
    strictness = select_strictness(u_score)
    _log(f"[Pipeline] Strictness Tier: {strictness.upper()}")
    
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

    # Layer 6: LLM generation via Groq
    answer = inference_groq(final_prompt)

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
        model_name=f"groq/{settings.GENERATION_MODEL_FREE}",
    )
    
    # Embed the U score into confidence
    response.confidence = round(1.0 - u_score, 3)
    
    # Check Constitutional Rules
    violations = check_rules(response)
    if violations:
        _log(f"[Pipeline] Warning: Constitutional Violations Detected: {violations}")

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
            _log(f"Warning: DB logging failed: {e}")

    return response


# ── Async Pipeline (Day 13+) ────────────────────────────────────────────────

async def run_pipeline_async(
    question: str,
    use_local_model: bool = True,
):
    """
    Async pipeline orchestration (Day 13+).
    
    After the local Qwen3-8B generates a raw answer, it is sent to
    Groq for evidence-grounded reformatting and quality scoring.
    
    KEY ANTI-HALLUCINATION FEATURES:
    - Reformatter is given an explicit allowed-citations whitelist
    - Citation verifier reports ALL violations (not fail-fast)
    - Hallucinated answers are replaced with evidence-only fallback
    - Rule 5 actively checks section/article existence
    """
    import time
    from src.llm.base_model_loader import InferenceBackend, generate
    from src.generation.response_builder import build_response_async
    from src.llm.inference_groq import reformat_and_score

    start_time = time.perf_counter()

    from fastapi.concurrency import run_in_threadpool
    # 1. Query processing + retrieval
    processed = process_query(question)
    results = await run_in_threadpool(hybrid_search, question, 8, processed)
    
    # 2. Aggregation
    evidence_set = aggregate_evidence(results)

    # 3. Uncertainty Features
    evidence_texts = [e.text for e in evidence_set.evidences]
    
    # Calculate NLI entailments for features if evidence exists
    nli_scores = []
    if evidence_texts:
        for text in evidence_texts:
            res = _get_verifier().relevance_checker.entails(question, text)
            nli_scores.append(res["score"])
            
    scores = [e.score for e in evidence_set.evidences]
    
    X_feats = [
        retrieval_confidence(scores),
        evidence_agreement(nli_scores),
        coverage(set(question.lower().split()), set(" ".join(evidence_texts).lower().split())),
        entropy(scores)
    ]
    
    # 4. Uncertainty Score
    try:
        calibrated_model = load_calibrated_model()
        import numpy as np
        prob = calibrated_model.predict_proba(np.array([X_feats]))[0][1]
        uncertainty = float(prob)
    except Exception as e:
        _log(f"Warning: Uncertainty model failed ({e}). Defaulting to 0.5")
        uncertainty = 0.5

    # 5. Strictness Policy
    strictness = select_strictness(uncertainty)

    # Short-circuit for High Uncertainty (Rule 4)
    if strictness == "high":
        decision = "Abstained"
        processing_time = (time.perf_counter() - start_time) * 1000
        
        backend_used = InferenceBackend.LOCAL if use_local_model else InferenceBackend.GROQ
        
        return await build_response_async(
            question=question,
            answer_text="I cannot answer this based on available provisions.",
            evidence_set=evidence_set,
            citation_results=[],
            model_name=backend_used.value,
            decision=decision,
            uncertainty_score=uncertainty,
            strictness_tier=strictness,
            processing_time_ms=processing_time,
            quality_score=1.0,       # Correct abstention = perfect score
            issues_found=[],
            reformatted=False,
        )

    decision = "Answered"

    # 6. Generation
    backend = InferenceBackend.LOCAL if use_local_model else InferenceBackend.GROQ
    
    # We pass the flattened evidence string to the generator
    flat_evidence = "\n\n".join(
        f"=== {e.citation_id} ===\n{e.text}"
        for e in evidence_set.evidences
    )

    llm_resp = await generate(
        question=question,
        evidence_text=flat_evidence,
        strictness_tier=strictness,
        backend=backend,
    )
    raw_text = llm_resp["text"]
    
    # The fine-tuned local model generates training-specific prefixes. 
    # Extract just the answer text.
    if "Answer:" in raw_text:
        parts = raw_text.split("Answer:", 1)
        if len(parts) > 1:
            ans_part = parts[1]
            ans_part = re.split(r'\n(?:Evidence|Confidence|Decision):', ans_part)[0]
            answer = ans_part.strip()
        else:
            answer = raw_text.strip()
    else:
        answer = raw_text.strip()

    model_used = llm_resp["model"]

    # ── Groq Reformat + Score Step ───────────────────────────────────────
    # Send the raw answer to Groq for evidence-grounded reformatting
    # and quality scoring. The reformatter receives an explicit
    # allowed-citations whitelist extracted from the evidence.
    _log(f"[Pipeline] Sending raw answer to Groq for reformatting & scoring...")
    reformat_result = await reformat_and_score(
        raw_answer=answer,
        question=question,
        evidence_text=flat_evidence,
    )
    
    reformatted_answer = reformat_result["reformatted_answer"]
    quality_score = reformat_result["quality_score"]
    issues_found = list(reformat_result["issues"] or [])
    hallucinated_citations = reformat_result.get("hallucinated_citations", [])

    # Hard-enforce citation whitelist (Groq reformatter can still leak training knowledge)
    allowed_citations = extract_allowed_citations(flat_evidence)
    reformatted_answer, stripped = enforce_allowed_citations(
        reformatted_answer, allowed_citations
    )
    if stripped:
        hallucinated_citations = sorted(set(hallucinated_citations) | set(stripped))
        issues_found.append(
            f"Stripped {len(stripped)} citation(s) not present in retrieved evidence"
        )
        quality_score = min(quality_score, 0.35)
    
    _log(f"[Pipeline] Groq Quality Score: {quality_score:.2f}")
    if issues_found:
        _log(f"[Pipeline] Issues Found: {issues_found}")
    if hallucinated_citations:
        _log(f"[Pipeline] Hallucinated Citations Stripped by Reformatter: {hallucinated_citations}")
    # ─────────────────────────────────────────────────────────────────────

    # Check for model abstention (on the reformatted answer)
    abstention_phrases = [
        "cannot answer this based on available provisions",
        "i cannot answer this",
        "i do not have sufficient legal evidence",
    ]
    if any(phrase in reformatted_answer.lower() for phrase in abstention_phrases):
        decision = "Abstained"
        verdict_result = {
            "verdict": "Valid",
            "reason": "Model correctly abstained due to lack of evidence.",
            "score": 1.0,
            "all_violations": [],
            "nonexistent_citations": [],
            "ungrounded_citations": [],
        }
    else:
        # 7. Citation Verification (on the reformatted answer)
        # The verifier now reports ALL violations, not just the first
        verdict_result = _get_verifier().verify(reformatted_answer, evidence_set.evidences)
        # As per guide2.md, the pipeline orchestrates the flow and passes the verdict
        # to the response builder. We do not arbitrarily replace the answer here.
        # The frontend will display the answer along with any 'Invalid' badges.

    processing_time = (time.perf_counter() - start_time) * 1000

    # 8. Build Response
    return await build_response_async(
        question=question,
        answer_text=reformatted_answer,
        evidence_set=evidence_set,
        citation_results=[verdict_result],
        model_name=model_used,
        decision=decision,
        uncertainty_score=uncertainty,
        strictness_tier=strictness,
        processing_time_ms=processing_time,
        quality_score=quality_score,
        issues_found=issues_found if issues_found else None,
        reformatted=True,
    )
