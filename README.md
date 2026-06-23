# Adaptive Constitutional RAG for Indian Tax Law

This repository implements the **Adaptive Constitutional RAG** framework, a sophisticated, uncertainty-calibrated Retrieval-Augmented Generation system designed for complex legal domains, specifically Indian Tax Law and the Constitution of India.

It mitigates hallucinations in legal Language Models by deploying an adaptive constitutional engine that dynamically tightens generation rules based on an intelligent **Uncertainty Estimation Module (Layer 4)**.

---

## 🏛️ System Architecture

The architecture consists of an 8-layer pipeline:

1. **Query Interface:** API (FastAPI) and Frontend (Next.js) entrypoints.
2. **Hybrid Legal Retrieval Engine:** Uses Qdrant's Reciprocal Rank Fusion (RRF) to combine dense embeddings (`BAAI/bge-m3`) with sparse exact-matching (BM25 via `FastEmbed`).
3. **Evidence Aggregation:** Dedupes overlapping legal chunks and groups them logically.
4. **Uncertainty Estimation Module (Layer 4):** A calibrated Gradient Boosting Classifier trained on four features (retrieval confidence, NLI evidence agreement, coverage, and entropy) to predict the likelihood of hallucination ($U \in [0, 1]$).
5. **Adaptive Constitutional Engine:** Enforces rules dynamically based on the $U$ score:
   - **Low $U$:** Detailed response allowed.
   - **Medium $U$:** Answer with caution; conflicts flagged.
   - **High $U$:** Force an abstention to prevent fabricated legal claims.
6. **Fine-Tuned Legal LLM:** (Target: `Qwen3-8B-Instruct` via QLoRA).
7. **Citation Verification:** Three-tiered verification:
   - *Existence Check:* Validates against a ground-truth JSON index.
   - *Grounding Check:* Verifies the citation was present in retrieved evidence.
   - *Relevance Check:* NLI entailment via a cross-encoder (`cross-encoder/nli-deberta-v3-base`).
8. **Response Generator:** Assembles the final structured output (Answer, Evidence, Section/Article references, Confidence, Decision).

---

## 🛠️ Technology Stack

- **Vector Database:** Qdrant (Self-hosted via Docker)
- **Relational Database:** PostgreSQL (for query/evaluation logging)
- **Dense/Sparse Embeddings:** `BAAI/bge-m3` + `Qdrant/bm25`
- **NLI Cross-Encoder:** `cross-encoder/nli-deberta-v3-base`
- **LLM Inferencing:** OpenRouter (`openai/gpt-oss-120b:free` and `gpt-oss-20b:free`)
- **Machine Learning (Uncertainty):** `scikit-learn` (Gradient Boosting, Platt Scaling)
- **Testing:** `pytest`

---

## 🚀 Getting Started

### 1. Environment Setup

Copy `.env.example` to `.env` and fill in your OpenRouter API key.

```bash
cp .env.example .env
```

**Note:** The OpenRouter API key is *never* hardcoded in the codebase. It is strictly injected into `src/config.py` via `pydantic-settings` from the `.env` file.

### 2. Infrastructure

Spin up the Qdrant, PostgreSQL, and Redis containers:

```bash
docker compose up -d qdrant postgres redis
```

Install Python dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Pipeline Execution (Days 1 to 10 completed)

The system is currently implemented and tested up to **Day 10** of the build roadmap.

**To run the QA Dataset Generation (Days 8 & 9):**
```bash
python -m scripts.gen_factoid
python -m scripts.gen_reasoning
python -m scripts.gen_multihop
python -m scripts.gen_unanswerable
python -m scripts.gen_adversarial
python -m finetuning.prepare_dataset
```
*This constructs a robust dataset formatted for ChatML, including Adversarial examples that inject fabricated laws (e.g., Section 999Z).*

**To train the Uncertainty Estimation Module (Day 10):**
```bash
python -m src.uncertainty.train_uncertainty_model
```
*This extracts features from the training dataset, fits a `GradientBoostingClassifier`, applies Platt scaling, and saves `calibrated_uncertainty_model.joblib`.*

---

## 🧪 Testing

The repository contains a robust suite of `pytest` integration and unit tests covering:
- **Retrieval Engine (`test_retrieval.py`):** Mocks and validates the hybrid search implementation across 15 legal queries.
- **Citation Verification (`test_citation.py`):** Validates regex extraction, ground-truth matching, and the cross-encoder logic.
- **Evidence Aggregation (`test_evidence.py`):** Ensures `ScoredPoint` adapters and chunk deduplication behave correctly.
- **LLM Inferencing (`test_llm.py`):** Validates prompt templating and external client behavior.
- **Uncertainty Module (`test_uncertainty.py`):** Validates the mathematical correctness of `retrieval_confidence`, `entropy`, `coverage`, and `evidence_agreement` functions.
- **Full Pipeline (`test_full.py`):** Verifies the orchestration sequence of the entire RAG flow.

**Run the tests:**
```bash
pytest tests/ -v
```

---

## 📖 Current State

Everything up to **Day 10** is fully functional, tested, and passing. The system is ready for the integration of the **Adaptive Constitutional Engine (Day 11)** and **QLoRA Fine-Tuning (Day 12)**.