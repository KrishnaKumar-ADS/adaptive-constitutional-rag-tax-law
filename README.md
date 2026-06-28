# Adaptive Constitutional RAG for Indian Tax Law

This repository implements the **Adaptive Constitutional RAG** framework, a sophisticated, uncertainty-calibrated Retrieval-Augmented Generation system designed for complex legal domains, specifically Indian Tax Law and the Constitution of India.

It mitigates hallucinations in legal Language Models by deploying an adaptive constitutional engine that dynamically tightens generation rules based on an intelligent **Uncertainty Estimation Module**.

---

## 🌟 Novelty of this Project

Unlike standard RAG systems that blindly retrieve context and generate answers, this project introduces three novel mechanisms designed specifically for high-stakes legal domains:

1. **Adaptive Strictness via Calibrated Uncertainty ($U$)**: 
   Instead of using a static prompt, the system evaluates the query's complexity, retrieval quality, and evidence agreement *before* generation. A machine-learning classifier (Gradient Boosting) predicts a calibrated uncertainty score ($U \in [0, 1]$). Based on this score, the **Constitutional Policy Engine** dynamically routes the query to different strictness tiers. If uncertainty is too high, the system forces a hard abstention ("I cannot answer this"), preventing fabricated legal claims before they happen.
2. **Zero-LLM Citation Verification**:
   The system does not trust the LLM to verify its own citations. It employs a three-tiered deterministic post-generation check:
   - *Existence Check*: Pure dictionary lookup against a ground-truth JSON index of the Income Tax Act and Constitution.
   - *Grounding Check*: Verifies the exact citation was actually present in the retrieved chunks.
   - *Relevance Check*: Uses a lightweight NLI cross-encoder model to mathematically prove the retrieved text entails the claim made by the LLM.
3. **Domain-Specific Fine-Tuning (QLoRA + DoRA)**:
   Generic models often hallucinate complex legal sub-sections. We overcome this by fine-tuning a custom `Qwen3-8B-Instruct` model specifically on an Indian Tax Law Q&A dataset. The fine-tuning explicitly bakes in abstention behavior for unanswerable queries and adversarial (fake-law) injections.

---

## 🏛️ System Architecture

The architecture consists of an end-to-end 8-layer pipeline:

1. **Query Interface:** API (FastAPI) and Frontend (Next.js) entrypoints.
2. **Hybrid Legal Retrieval Engine:** Uses Qdrant's Reciprocal Rank Fusion (RRF) to combine dense embeddings (`BAAI/bge-m3`) with sparse exact-matching (BM25 via `FastEmbed`).
3. **Evidence Aggregation:** Dedupes overlapping legal chunks and groups them logically.
4. **Uncertainty Estimation Module:** A calibrated Gradient Boosting Classifier trained on four features:
   - `retrieval_confidence` (Fusion scores)
   - `evidence_agreement` (NLI overlap)
   - `coverage` (Keyword match density)
   - `entropy` (Score distribution)
5. **Adaptive Constitutional Engine:** Enforces rules dynamically based on the $U$ score:
   - **Low $U$ (< 0.2):** Standard generation allowed.
   - **Medium $U$ (0.2 - 0.7):** Answer with caution; conflicts flagged.
   - **High $U$ (> 0.7):** Force abstention.
6. **Fine-Tuned Legal LLM:** The local specialized Qwen model processes the prompt and evidence.
7. **Citation Verification:** The Zero-LLM deterministic verifier runs on the output.
8. **Response Builder:** Assembles the final structured output (Answer, Evidence, Section/Article references, Confidence, Validity Badges) and passes it to the frontend.

---

## 🗂️ Project Structure

```text
├── api/                   # FastAPI backend service
│   ├── routers/           # API Endpoints (query, health)
│   ├── schemas.py         # Pydantic schemas for API requests
│   └── main.py            # Application entrypoint
├── benchmark/             # Evaluation harness
│   ├── baselines/         # 5 baseline implementations for comparison
│   └── metrics/           # Hallucination rate, LGS, Calibration error
├── data/                  # Datasets and embeddings
│   ├── processed/         # Chunks, JSON indices, fine-tuning data
│   ├── qa_dataset/        # Factoid, Reasoning, Multi-hop, Adversarial
│   └── raw/               # Original PDFs (Income Tax Act, Constitution)
├── docs/                  # Architecture and dataset documentation
├── finetuning/            # Scripts for training the local LLM
│   ├── qlora_config.yaml  # Hyperparameters for QLoRA
│   ├── train_qlora.py     # Unsloth + TRL training script
│   └── merge_adapter.py   # Merging LoRA into base weights
├── frontend/              # Next.js React Application
│   ├── app/               # Main pages
│   └── components/        # CitationCards, EvidencePanels, ConfidenceBadges
├── scripts/               # Utility scripts (infra check, indexing)
├── src/                   # Core RAG Application Logic
│   ├── citation/          # Verifier, Section/Article validators, Grounding
│   ├── constitutional/    # Rules engine, Enforcement Policy
│   ├── evidence/          # Aggregator
│   ├── generation/        # Output schema, Response Builder
│   ├── ingestion/         # PDF parser, chunker, metadata extractor
│   ├── llm/               # Local inference, Groq/OpenRouter fallback
│   ├── retrieval/         # Dense, Sparse, Hybrid search, Reranker
│   └── uncertainty/       # Feature extraction, classifier, calibration
├── tests/                 # Pytest suite
├── .env                   # Environment variables
├── docker-compose.yml     # Infrastructure (Qdrant, Postgres, Redis, API)
└── README.md              # You are here
```

---

## 🛠️ Technology Stack

- **Backend:** FastAPI, Python 3.11, SQLAlchemy
- **Frontend:** Next.js (React), TypeScript, TailwindCSS
- **Vector Database:** Qdrant (Self-hosted via Docker)
- **Relational Database:** PostgreSQL (for query/evaluation logging)
- **Document Parsing:** Docling (for precise Section/Article extraction)
- **Machine Learning (Uncertainty):** `scikit-learn` (Gradient Boosting, Platt Scaling)
- **Testing:** `pytest`

---

## 🧠 Models Used

1. **Generation (Local/Fine-Tuned):** `Qwen3-8B-Instruct`
   - *Purpose:* Core legal reasoning and answer generation.
2. **Dense Embeddings:** `BAAI/bge-m3`
   - *Purpose:* High-quality semantic search across legal texts.
3. **Sparse Embeddings:** `Qdrant/bm25` (via `FastEmbed`)
   - *Purpose:* Exact lexical matching for specific Section and Article numbers.
4. **NLI Cross-Encoder:** `cross-encoder/nli-deberta-v3-base`
   - *Purpose:* Verifies that the retrieved evidence actually entails the LLM's claims.

---

## 🔬 Fine-Tuning Process in Detail

To ensure the model fundamentally understands Indian Tax Law and knows *when to abstain*, we conducted a highly specialized fine-tuning process.

### 1. Dataset Synthesis
We algorithmically generated a rigorous dataset of ~700-950 QA pairs across 5 categories:
- **Factoid**: Direct definitions (e.g., "What is Section 10(1)?").
- **Reasoning**: Interpretation of exceptions and caveats.
- **Multi-hop**: Connecting the Income Tax Act to its constitutional origin (e.g., Article 265).
- **Unanswerable**: Topics absent from the corpus, explicitly teaching the model to abstain.
- **Adversarial**: Injecting fake laws (`Section 999Z`) to teach the model to reject fabricated premises.

### 2. QLoRA and DoRA Configuration
We used **Unsloth** and **bitsandbytes** for accelerated, memory-efficient training on a single NVIDIA A100 GPU.
- **Base Model:** `Qwen/Qwen3-8B-Instruct`
- **Quantization:** The base model is loaded in 4-bit precision using NormalFloat4 (NF4) quantization. We utilize `bfloat16` for compute dtype and enable double quantization to maximize VRAM efficiency without sacrificing accuracy.
- **Method:** Quantized Low-Rank Adaptation (QLoRA) combined with Weight-Decomposed Low-Rank Adaptation (DoRA) for enhanced learning capacity.
- **Parameters:** `r = 64`, `lora_alpha = 128`, targeting `all-linear` modules (Q, K, V, O, Gate, Up, Down), with a dropout of `0.05`.

### 3. Training
- **Framework:** `TRL SFTTrainer` via Hugging Face.
- **Hyperparameters:** Cosine learning rate scheduler, 3 epochs, optimized with `paged_adamw_8bit`.
- **Formatting:** Data was structured using the ChatML template, explicitly including citation brackets so the model learns the exact format the pipeline expects.

### 4. Merging and Export
Upon completion and evaluation on a held-out split, the LoRA adapter was merged into the full 16-bit weights (`save_pretrained_merged`). Finally, the model was exported to **GGUF format (Q4_K_M)** for rapid, CPU/GPU-agnostic local inference via Ollama.

---

## 🚀 Getting Started

### 1. Environment Setup

Copy `.env.example` to `.env` and fill in any required keys.

```bash
cp .env.example .env
```

### 2. Infrastructure

Spin up the Qdrant, PostgreSQL, and Redis containers:

```bash
docker compose up -d qdrant postgres redis
```

Install Python dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Start the Application

The system uses a FastAPI backend and a Next.js frontend. Start both to test the pipeline:

**Backend:**
```bash
.\venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

**Frontend:**
```bash
cd frontend
npm run dev
```

Visit `http://localhost:3000` to interact with the Adaptive Constitutional RAG interface!