# 15-Day Build Roadmap — Adaptive Constitutional RAG for Indian Tax Law

> Companion to `guide.md`. That file tells you **what** to build and **why**. This file tells you **what to do today**.
> Each day = one focused build session (assume ~4–6 hours). Each day ends with something that *runs*, not just files that exist.
> Hard dependency rule: **Day 9's dataset must be finished before Day 10 (uncertainty training) and Day 12 (fine-tuning)** — everything downstream consumes it. Don't let Days 8–9 slip.

---

## Day 0 (Prep — do this before Day 1, ~1 hour)

Get every account/credential sorted now so no day gets blocked mid-session waiting on a signup email.

- [ ] **OpenRouter account** → generate an API key → add **$10 of credits once** (raises free-tier cap from 50/day to 1,000/day forever — you will need this by Day 14)
- [ ] **Hugging Face account** → accept the license/gate on `Qwen/Qwen3-8B-Instruct` if gated, generate an access token
- [ ] **Google Colab** access confirmed (free T4 is enough; use your existing A100-capable account from VLDIS if you have Colab Pro)
- [ ] **Weights & Biases** account (free academic tier — use your IIIT Sricity email)
- [ ] **Docker Desktop / Docker Engine** installed and running locally
- [ ] Source PDFs downloaded: **Income Tax Act, 1961** (consolidated/bare act PDF) and **Constitution of India** → drop into `data/raw/` once the repo exists on Day 1
- [ ] `git init` the repo, create a GitHub remote (you'll want commit history per day for the reproducibility section of the paper)

---

## Day 1 — Project Scaffolding & Infra

**🎯 Goal:** Empty-but-running infrastructure. No business logic yet — just prove every service talks to every other service.

**Tasks:**
1. Create the full directory skeleton from `guide.md` Section 5 (`mkdir -p` every folder — `src/`, `data/`, `finetuning/`, `benchmark/`, `api/`, `frontend/`, `notebooks/`, `tests/`, `scripts/`, `docs/`).
2. Write `requirements.txt`, `pyproject.toml`, `.gitignore`, `.env.example` (copy from `guide.md` Section 4), then copy `.env.example` → `.env` and fill in your real `OPENROUTER_API_KEY`.
3. Write `docker-compose.yml` (Qdrant + Postgres + Redis only today — leave the `api` service commented out, you have no app yet).
4. Create a Python 3.11 virtualenv, `pip install -r requirements.txt`.
5. Write `src/config.py` — a `pydantic-settings` class that loads everything from `.env` (model names, thresholds, DB URLs).
6. `docker compose up -d qdrant postgres redis`. Write a 10-line throwaway script `scripts/check_infra.py` that pings all three: `QdrantClient(...).get_collections()`, a raw `psycopg2.connect(...)`, and `redis.Redis.from_url(...).ping()`.
7. Initialize Postgres schema: write `src/db/models.py` (SQLAlchemy) with at minimum a `queries` table (id, question, evidence_ids, uncertainty_score, decision, citations_json, created_at) — you'll insert into this from Day 7 onward. Run an Alembic migration to create it.
8. Commit: `git commit -m "Day 1: infra scaffolding"`.

**📁 Files touched:** `requirements.txt`, `pyproject.toml`, `.gitignore`, `.env`, `.env.example`, `docker-compose.yml`, `src/config.py`, `src/db/models.py`, `alembic/`, `scripts/check_infra.py`

**🧪 Validate:** `python scripts/check_infra.py` prints `Qdrant OK / Postgres OK / Redis OK`.

**🏁 End-of-day deliverable:** `docker compose up -d` brings up a clean 3-service stack; your Python env can talk to all three; `queries` table exists in Postgres.

---

## Day 2 — Document Ingestion Pipeline

**🎯 Goal:** Raw PDFs → clean, chunked, metadata-tagged text on disk. No vector DB writes yet — get the data clean first.

**Tasks:**
1. Drop `income_tax_act_1961.pdf` and `constitution_of_india.pdf` into `data/raw/`.
2. Write `src/ingestion/pdf_parser.py` using Docling (preferred — handles the numbered sub-section structure and tabular schedules in the Act far better than naive `pdfplumber` text extraction). Output: a list of `{page, raw_text}` per document.
3. Write `src/ingestion/chunker.py` — chunk into 512–1024 token windows. **Do not chunk blindly by character count** — split on section/article boundaries first (regex for `Section \d+[A-Z]*`, `Article \d+`), then sub-chunk anything still over 1024 tokens. This matters enormously for citation accuracy later.
4. Write `src/ingestion/metadata_extractor.py` — tags every chunk with `{source: "income_tax_act"|"constitution", section_number, article_number, title, page}`. This is the single most important file for Layer 7 later — get the regex right now and you save yourself debugging time on Day 6.
5. Write `src/ingestion/run_ingestion.py` — CLI: `python -m src.ingestion.run_ingestion` runs parser → chunker → metadata_extractor → writes `data/processed/chunks.jsonl`.
6. Manually spot-check 20 random lines of `chunks.jsonl` — confirm section numbers are correctly attached and no chunk straddles two unrelated sections.

**📁 Files touched:** `src/ingestion/pdf_parser.py`, `chunker.py`, `metadata_extractor.py`, `run_ingestion.py`, `data/raw/*.pdf`, `data/processed/chunks.jsonl`

**🧪 Validate:** `wc -l data/processed/chunks.jsonl` shows a sane number of chunks (expect roughly 400–900 for the Act + Constitution combined); every line has non-null `source` and at least one of `section_number`/`article_number`.

**🏁 End-of-day deliverable:** A clean `chunks.jsonl` you'd trust enough to hand to someone else's pipeline.

---

## Day 3 — Ground-Truth Index + Vector DB Population

**🎯 Goal:** Qdrant is populated with embedded chunks (dense + sparse), and you have a ground-truth lookup table for citation verification.

**Tasks:**
1. Write `data/processed/section_index.json` and `article_index.json` generation logic (a small script reading `chunks.jsonl` and grouping by `section_number`/`article_number` → `{"Section 10(1)": {"text": ..., "page": ..., "chunk_ids": [...]}}`). This is your **Layer 7 ground truth** — every citation existence check on Day 6 hits these files directly, no LLM involved.
2. Install/pull `BAAI/bge-m3` locally (via `FlagEmbedding`) — test it embeds a sample sentence and returns both `dense_vecs` and `lexical_weights` (sparse).
3. In `scripts/seed_vector_db.py`: create the Qdrant collection `tax_law` with **named vectors** — `dense` (1024-dim, cosine) and `sparse` (IDF-modified) — exactly as shown in `guide.md` Section 8.
4. Loop `chunks.jsonl` in batches, embed with `bge-m3`, upsert into Qdrant with both vector types + the full metadata payload.
5. Run `python scripts/seed_vector_db.py`.

**📁 Files touched:** `scripts/build_citation_index.py` (new, generates the two index JSONs), `data/processed/section_index.json`, `article_index.json`, `scripts/seed_vector_db.py`

**🧪 Validate:** `curl http://localhost:6333/collections/tax_law` shows `points_count` matching your chunk count. `json.load(open("data/processed/section_index.json"))["Section 10(1)"]` returns real text.

**🏁 End-of-day deliverable:** A fully populated, queryable Qdrant collection, plus the JSON ground-truth files Layer 7 will depend on for the rest of the project.

---

## Day 4 — Hybrid Retrieval Engine (Layer 2)

**🎯 Goal:** Working dense + sparse + fused retrieval, with tests proving it actually finds the right law.

**Tasks:**
1. Write `src/retrieval/dense_retriever.py` — dense-only Qdrant query function.
2. Write `src/retrieval/sparse_retriever.py` — sparse-only (BM25 via `Qdrant/bm25` FastEmbed) query function.
3. Write `src/retrieval/hybrid_retriever.py` — the fused `Prefetch` + `FusionQuery(fusion=models.Fusion.RRF)` query from `guide.md` Section 8. Return Top-K (start with K=8) scored passages with full metadata.
4. (Optional but recommended) Write `src/retrieval/reranker.py` using `bge-reranker-v2-m3` as a cross-encoder rerank pass on the fused Top-20 → Top-8.
5. Write `tests/test_retrieval.py` with **at least 10 hand-written test queries** with known-correct expected sections, e.g.:
   - "Is agricultural income taxable?" → expect `Section 10(1)` in top 3
   - "What does Article 265 say about taxation?" → expect `Article 265` in top 3
6. Run the tests, debug chunking/metadata issues from Day 2 if retrieval is missing obvious answers (this is normal — expect to go back and fix a regex in `metadata_extractor.py`).

**📁 Files touched:** `src/retrieval/dense_retriever.py`, `sparse_retriever.py`, `hybrid_retriever.py`, `reranker.py`, `tests/test_retrieval.py`

**🧪 Validate:** `pytest tests/test_retrieval.py -v` — aim for ≥8/10 passing today; perfect retrieval isn't required yet, but obvious misses need fixing now, not on Day 7.

**🏁 End-of-day deliverable:** A `hybrid_search(query)` function you trust, backed by passing tests.

---

## Day 5 — Evidence Aggregation (Layer 3) + Rate-Limited LLM Client

**🎯 Goal:** Clean, deduplicated evidence sets, and a safe, reusable way to call free OpenRouter models without hitting 429s.

**Tasks:**
1. Write `src/evidence/evidence_aggregator.py` — takes hybrid retrieval's Top-K, dedupes overlapping/adjacent chunks from the same section, groups by source type (tax section vs. constitutional article), and outputs a single clean `EvidenceSet` object (list of `{citation_id, text, source_type, score}`).
2. Write `src/llm/inference_openrouter.py` — an async client wrapping the OpenAI-compatible endpoint, with an `asyncio.Semaphore(20)`-style limiter respecting the **20 requests/minute** cap, plus exponential backoff on 429s. Default model from `settings.GENERATION_MODEL_FREE` (`openai/gpt-oss-120b:free`), with a `fast=True` flag to route to `settings.FAST_MODEL_FREE` (`openai/gpt-oss-20b:free`).
3. Write `src/llm/prompts.py` — shared prompt-building helpers (system prompt skeleton, evidence-injection formatting).
4. Smoke-test: call `inference_openrouter.py` with a hardcoded evidence set + question, confirm you get a coherent response back and the rate limiter doesn't choke on a burst of 5 rapid calls.

**📁 Files touched:** `src/evidence/evidence_aggregator.py`, `src/llm/inference_openrouter.py`, `src/llm/prompts.py`

**🧪 Validate:** A quick script firing 25 rapid requests completes without a single unhandled 429 (the limiter should queue/backoff gracefully, not crash).

**🏁 End-of-day deliverable:** Evidence aggregation + a production-safe LLM client you'll reuse for the rest of the project (dataset generation, baselines, evaluation).

---

## Day 6 — Citation Verification Module (Layer 7)

**🎯 Goal:** A citation verifier that can immediately catch fake law — build and test this **before** you wire up generation, so you trust it by Day 7.

**Tasks:**
1. Write `src/citation/section_validator.py` and `article_validator.py` — regex-extract any `Section X`/`Article Y` mentioned in a claim, look up against `section_index.json`/`article_index.json` from Day 3. Pure dictionary lookup, **no LLM call**.
2. Write `src/citation/grounding_checker.py` — given a citation + the `EvidenceSet` actually used for this query, check the citation was genuinely retrieved (not just real, but *used*).
3. Write `src/citation/relevance_checker.py` — NLI entailment check. Use `sentence-transformers` with `cross-encoder/nli-deberta-v3-base` locally (free, fast, no API call) to test whether the cited passage actually entails the claim made about it.
4. Write `src/citation/citation_verifier.py` orchestrating all three → returns `Valid` / `Invalid` / `Partially Supported` exactly as specified in `guide.md` Section 11.
5. Write `tests/test_citation_verifier.py` with explicit cases: a real, correctly-used citation → `Valid`; a fabricated `Section 999Z` → `Invalid`; a real section cited but never retrieved → `Partially Supported`.

**📁 Files touched:** `src/citation/section_validator.py`, `article_validator.py`, `grounding_checker.py`, `relevance_checker.py`, `citation_verifier.py`, `tests/test_citation_verifier.py`

**🧪 Validate:** `pytest tests/test_citation_verifier.py -v` — all cases pass, including the fake-section rejection.

**🏁 End-of-day deliverable:** A citation verifier you can trust to catch hallucinated law with zero LLM calls involved in the existence check.

---

## Day 7 — First End-to-End Demo: Baseline C (Standard RAG)

**🎯 Goal:** Type a real tax-law question, get back a structured, citation-checked answer. This is your first genuine milestone — everything before today was plumbing.

**Tasks:**
1. Write `src/generation/output_schema.py` — the `LegalResponse` Pydantic model from `guide.md` Section 12.
2. Write `src/generation/response_builder.py` — assembles the final structured response from: the generated answer text, the evidence set, and the citation verdicts.
3. Write `benchmark/baselines/baseline_c_standard_rag.py` — wires: `hybrid_search()` → `evidence_aggregator` → `inference_openrouter()` (with a prompt instructing the model to cite sections/articles inline) → `citation_verifier()` → `response_builder()`.
4. Write a tiny `scripts/ask.py` CLI: `python scripts/ask.py "Can agricultural income be taxed?"` → prints the full structured response.
5. Run it on 5–10 questions manually. Note where citations come back `Invalid`/`Partially Supported` even though the model *should* have gotten it right — these are your first qualitative observations for the paper's case-analysis section later.
6. Insert each query's results into the `queries` Postgres table (id, question, evidence_ids, citations_json, raw_response) — start building your evaluation log now, not later.

**📁 Files touched:** `src/generation/output_schema.py`, `response_builder.py`, `benchmark/baselines/baseline_c_standard_rag.py`, `scripts/ask.py`

**🧪 Validate:** `python scripts/ask.py "Is agricultural income exempt from tax?"` returns a full `LegalResponse` JSON with `Section 10(1)` validated `Valid`.

**🏁 End-of-day deliverable:** A working, demoable Baseline C — retrieval + generation + citation checking, end to end. Good moment to record a quick demo video/GIF for your README.

---

## Day 8 — Dataset Construction, Part 1: Factoid + Reasoning

**🎯 Goal:** The first two of five question categories, at 150–200 each.

**Tasks:**
1. Write a small generator script (`scripts/gen_factoid.py`) that loops `section_index.json`/`article_index.json` and template-generates factoid questions: *"What does Section 10(1) state?"*, *"What is Article 265?"* — zero LLM cost, deterministic, high precision. Target 150–200.
2. Write `scripts/gen_reasoning.py`: for a sample of sections, prompt `gpt-oss-120b:free` via your Day-5 client with *"Given this provision, generate one reasoning question whose answer requires interpreting (not just quoting) it, plus a model answer."* Target 150–200.
3. **Manually review a 20% random sample** of the reasoning questions for correctness — bad reasoning questions in training data will quietly poison your fine-tune on Day 12, so catch this now.
4. Write `data/qa_dataset/dataset_schema.md` documenting the exact JSONL schema you're using (`{question, answer, evidence_citations, category, difficulty}`) so every later category file is consistent.
5. Save outputs to `data/qa_dataset/factoid_questions.jsonl` and `reasoning_questions.jsonl`.

**📁 Files touched:** `scripts/gen_factoid.py`, `scripts/gen_reasoning.py`, `data/qa_dataset/factoid_questions.jsonl`, `reasoning_questions.jsonl`, `dataset_schema.md`

**🧪 Validate:** `wc -l` on both files shows 150+ lines each; every record validates against the schema (write a 5-line `jsonschema` check).

**🏁 End-of-day deliverable:** 300–400 quality-checked Q&A pairs across factoid + reasoning categories.

---

## Day 9 — Dataset Construction, Part 2: Multi-hop, Unanswerable, Adversarial + Train/Eval Split

**🎯 Goal:** Finish the QA dataset completely. **Nothing after today proceeds without this being done.**

**Tasks:**
1. Write `scripts/gen_multihop.py`: feed **two related passages** (a tax section + the constitutional article it derives from, e.g. Section 4 + Article 265) into `openrouter/owl-alpha` (long context handles multiple full sections comfortably) and ask for a question requiring both. Target 100–150.
2. Write `scripts/gen_unanswerable.py`: hand-craft or LLM-assist questions about topics genuinely absent from your corpus (e.g., a tax topic only covered in a CBDT circular you haven't ingested). Target 100. Label these explicitly so the citation verifier and uncertainty module both learn "no evidence → abstain."
3. Write `benchmark/fake_law_injection.py` and `scripts/gen_adversarial.py`: programmatically swap real section/article numbers in well-formed questions for fake ones (`Section 999Z`, `Article 420B`). Target 100.
4. Combine all five category files. Write `finetuning/prepare_dataset.py` — converts everything into the ChatML format from `guide.md` Section 7.1, then does an **80/20 train/eval split, stratified by category** so your held-out eval set has representatives from every category. Output `data/processed/finetune_train.jsonl` and `finetune_eval.jsonl`.
5. **Lock the eval split** — copy it somewhere immutable (or just never re-run the split script again) since this is what Day 14's evaluation sweep and your paper's results table depend on.

**📁 Files touched:** `scripts/gen_multihop.py`, `gen_unanswerable.py`, `gen_adversarial.py`, `benchmark/fake_law_injection.py`, `finetuning/prepare_dataset.py`, `data/qa_dataset/multi_hop_questions.jsonl`, `unanswerable_questions.jsonl`, `adversarial_questions.jsonl`, `data/processed/finetune_train.jsonl`, `finetune_eval.jsonl`

**🧪 Validate:** All five `data/qa_dataset/*.jsonl` files exist with target counts; `finetune_train.jsonl` + `finetune_eval.jsonl` together account for every record; every category appears in the eval split.

**🏁 End-of-day deliverable:** A complete, schema-consistent, quality-checked QA dataset (700–950 examples across 5 categories) with a locked held-out eval split. **This is the project's single most important artifact — treat today as a checkpoint, not a normal day.**

---

## Day 10 — Uncertainty Estimation Module (Layer 4)

**🎯 Goal:** A trained, calibrated uncertainty score `U ∈ [0,1]` — your core research contribution.

**Tasks:**
1. Write `src/uncertainty/features.py` — implement the four named features exactly as in `guide.md` Section 9: `retrieval_confidence()`, `evidence_agreement()`, `coverage()`, `entropy()`.
2. Generate labeled training data for the uncertainty model: run Baseline C (Day 7) over your full `finetune_train.jsonl` questions, capture the four features for each, and label each as hallucinated/not using your **citation verifier's `Invalid` verdicts + the `unanswerable`/`adversarial` category labels** as ground truth (you don't need manual labeling for most of this — your dataset categories already encode it).
3. Write `src/uncertainty/uncertainty_model.py` — a small classifier (logistic regression or `sklearn.ensemble.GradientBoostingClassifier`) trained on `(4 features) → hallucinated (0/1)`, exposing `predict_proba()` as your raw uncertainty score.
4. Write `src/uncertainty/train_uncertainty_model.py` — trains, evaluates on a held-out slice, saves the model (`joblib`).
5. Write `src/uncertainty/calibration.py` — apply temperature or Platt scaling on top so `U` is a genuinely calibrated probability. Plot a reliability diagram (predicted probability vs. observed frequency) and save it to `docs/` — you'll want this exact plot for the paper's Calibration Error result.

**📁 Files touched:** `src/uncertainty/features.py`, `uncertainty_model.py`, `train_uncertainty_model.py`, `calibration.py`, `data/processed/uncertainty_training_data.jsonl` (intermediate), saved model artifact

**🧪 Validate:** On the held-out eval split, mean `U` for `unanswerable`/`adversarial` questions is meaningfully higher than for `factoid` questions — if it isn't, your features aren't discriminative yet and need revisiting before moving on.

**🏁 End-of-day deliverable:** A calibrated uncertainty model that demonstrably separates "should abstain" from "safe to answer" cases.

---

## Day 11 — Adaptive Constitutional Engine (Layer 5) + Wire the Full (Pre-Fine-Tune) Pipeline

**🎯 Goal:** Uncertainty-driven behavior switching, and a single `pipeline.py` that runs Layers 1–5, 7, 8 end to end (Layer 6 still uses the free OpenRouter model — fine-tuning comes Day 12).

**Tasks:**
1. Write `src/constitutional/rules.py` — encode Rules 1–5 from the spec as explicit, checkable constraints (not just prose — e.g., Rule 2 "must cite source" becomes a programmatic check that `output_schema.section_references` is non-empty whenever `decision == "Answered"`).
2. Write `src/constitutional/enforcement_policy.py` — the `select_strictness(U)` threshold router from `guide.md` Section 10.
3. Write the three prompt templates in `src/constitutional/prompt_templates/` (`low_uncertainty.txt`, `medium_uncertainty.txt`, `high_uncertainty_abstain.txt`) — make the strictness language genuinely different, not cosmetically different, since your ablation depends on this having a measurable effect.
4. Write `src/pipeline.py` — orchestrates: query → `hybrid_retriever` → `evidence_aggregator` → `uncertainty` features + calibrated `U` → `enforcement_policy.select_strictness(U)` → load matching prompt template → `inference_openrouter()` (still the free model today) → `citation_verifier` → `response_builder`.
5. Manually test 3 questions designed to land in each tier (one obvious factoid, one borderline reasoning question, one adversarial/unanswerable) — confirm you see visibly different `decision` values (`Answered` / `Answered with Caveats` / `Abstained`).

**📁 Files touched:** `src/constitutional/rules.py`, `enforcement_policy.py`, `prompt_templates/*.txt`, `src/pipeline.py`

**🧪 Validate:** The three test questions above produce three different `decision` outcomes as expected.

**🏁 End-of-day deliverable:** A complete, working 8-layer pipeline (minus your own fine-tuned model) — functionally this is Baseline D already.

---

## Day 12 — QLoRA Fine-Tuning

**🎯 Goal:** A trained LoRA adapter on `Qwen3-8B-Instruct` over your dataset, running on a free Colab GPU.

**Tasks:**
1. Write `finetuning/qlora_config.yaml` exactly as in `guide.md` Section 7.2 (r=16, DoRA on, `all-linear`, cosine LR schedule, 3 epochs to start).
2. Write `finetuning/train_qlora.py` (Unsloth + TRL `SFTTrainer`, as in `guide.md` Section 7.3).
3. Open `notebooks/02_qlora_finetuning_colab.ipynb`, mount your repo (clone from GitHub or upload), point it at `data/processed/finetune_train.jsonl` / `finetune_eval.jsonl` from Day 9, connect your W&B account, and **run training on a T4 (or your A100 if available)**.
4. Watch the eval loss curve in W&B — if it's still dropping at epoch 3, consider one more epoch; if it's overfitting (eval loss rising while train loss falls), stop early and reduce epochs in a re-run.
5. Save the LoRA adapter checkpoint back to your repo / Drive.

**📁 Files touched:** `finetuning/qlora_config.yaml`, `train_qlora.py`, `notebooks/02_qlora_finetuning_colab.ipynb`, `finetuning/checkpoints/` (adapter output)

**🧪 Validate:** Training completes without OOM; W&B shows a sane loss curve (decreasing, not diverging); adapter checkpoint files exist (`adapter_model.safetensors`, `adapter_config.json`).

**🏁 End-of-day deliverable:** A trained QLoRA adapter for Qwen3-8B-Instruct, specialized on your Indian tax-law dataset.

---

## Day 13 — Merge, Deploy, and Wire Layer 6 (Full Baseline E)

**🎯 Goal:** Your fine-tuned model is live inside the pipeline, and all five baselines (A–E) exist as independently runnable scripts.

**Tasks:**
1. Write `finetuning/merge_adapter.py` — merges the LoRA adapter into full 16-bit weights (`save_pretrained_merged`).
2. Write `src/llm/base_model_loader.py` and `src/llm/inference_local.py` — serve the merged model via **vLLM** if you have a GPU at inference time, or **Ollama** (after `export_gguf.py` quantized export) for CPU-friendly serving.
3. Update `src/pipeline.py` to call `inference_local.py` (your fine-tuned model) instead of `inference_openrouter.py` for the actual answer generation — keep the OpenRouter call as a config-switchable fallback.
4. Now write the remaining baseline scripts, since all the building blocks exist:
   - `benchmark/baselines/baseline_a_pure_llm.py` — no retrieval, no rules, raw `gpt-oss-120b:free` call
   - `benchmark/baselines/baseline_b_finetuned_llm.py` — your fine-tuned model, no retrieval
   - `benchmark/baselines/baseline_d_rag_static_rules.py` — retrieval + generation + **always-on** strictest rule set (no adaptivity — this is your key ablation comparison against E)
   - `benchmark/baselines/baseline_e_full_system.py` — calls `src/pipeline.py` directly, your complete system
5. Smoke-test all five baselines on the same 5 questions, side by side, and eyeball the differences.

**📁 Files touched:** `finetuning/merge_adapter.py`, `export_gguf.py`, `src/llm/base_model_loader.py`, `inference_local.py`, `src/pipeline.py` (updated), `benchmark/baselines/baseline_a_pure_llm.py`, `baseline_b_finetuned_llm.py`, `baseline_d_rag_static_rules.py`, `baseline_e_full_system.py`

**🧪 Validate:** All five baseline scripts run without error on a shared test question and visibly differ in behavior (A hallucinates more freely, E cites and abstains appropriately).

**🏁 End-of-day deliverable:** The complete system, all 5 baselines implemented — everything Section 9's experiments need now exists.

---

## Day 14 — Evaluation Harness: Metrics + Full Sweep

**🎯 Goal:** The results table for your paper.

**Tasks:**
1. Implement each metric file in `benchmark/metrics/`: `hallucination_rate.py`, `citation_precision_recall.py`, `calibration_error.py`, `abstention_accuracy.py`, `evidence_coverage.py`, and `legal_groundedness_score.py` (the `w1·EvidenceSupport + w2·CitationValidity + w3·ConstitutionalConsistency` formula from `guide.md` Section 14, starting at equal weights).
2. Write `benchmark/run_evaluation.py` — iterates all 5 baselines × your locked `finetune_eval.jsonl` (Day 9) × the 5 experiments from the spec (hallucination reduction, fake-law injection rejection, abstention quality, cross-document reasoning, citation accuracy), logging every result row to the `queries` Postgres table and to W&B.
3. **Before running the full sweep**, confirm your $10 OpenRouter credit (Day 0) is in effect — at 1,000 requests/day this sweep is a few hours; at 50/day it's not finishing today.
4. Run the full sweep. While it runs (this is largely unattended), use the time to start drafting `docs/architecture.md`.
5. Once complete, generate the comparison table (Baselines A–E × all metrics) and at least one ablation plot (uncertainty threshold vs. hallucination rate) in `notebooks/04_evaluation_analysis.ipynb`.

**📁 Files touched:** `benchmark/metrics/*.py`, `benchmark/run_evaluation.py`, `notebooks/04_evaluation_analysis.ipynb`

**🧪 Validate:** The sweep completes for all 5 baselines without unrecovered rate-limit failures; the results table shows Baseline E outperforming A–D on hallucination rate and LGS (if it doesn't, that's a real finding too — note it honestly for the paper rather than tuning thresholds until the numbers look right).

**🏁 End-of-day deliverable:** A complete results table and at least one ablation chart — the empirical core of your paper.

---

## Day 15 — API, Frontend, Docs, Final Polish

**🎯 Goal:** A demoable, documented, reproducible deliverable — `docker compose up` and it works.

**Tasks:**
1. Write `api/main.py`, `api/routers/query.py` (`POST /query` → calls `src/pipeline.py`), `health.py`, `admin_ingest.py`, `api/schemas.py`, `dependencies.py` (DB session, Qdrant client, model singletons loaded once at startup, not per-request).
2. Build the minimal `frontend/`: `app/page.tsx` with a query box, `EvidencePanel.tsx`, `ConfidenceBadge.tsx` (visualizing `U`/confidence), `CitationCard.tsx` (Valid/Invalid/Partial badges) — wire it to the FastAPI backend.
3. Uncomment and finish the `api` service in `docker-compose.yml`; run `docker compose up` from a clean checkout (delete local `__pycache__`/`node_modules` first) to genuinely test reproducibility.
4. Write `docs/architecture.md` (the 8-layer diagram + data flow — this becomes your paper's Methods section), `docs/dataset_card.md` (documents exactly how the Day 8–9 dataset was built — required for IEEE reproducibility), and finish `README.md` (setup instructions, architecture summary, results snapshot).
5. Run `tests/test_pipeline_e2e.py` one final time as a full regression check.
6. Tag a release commit: `git tag v1.0 && git push --tags`.

**📁 Files touched:** `api/main.py`, `routers/*.py`, `schemas.py`, `dependencies.py`, `frontend/app/page.tsx`, `components/*.tsx`, `docker-compose.yml` (finalized), `docs/architecture.md`, `dataset_card.md`, `README.md`, `tests/test_pipeline_e2e.py`

**🧪 Validate:** A clean `git clone` + `docker compose up` (with `.env` filled in) gives a working demo on a machine that's never seen this repo before.

**🏁 End-of-day deliverable:** The finished system — working demo, full documentation, results table, ready to move straight into writing the paper draft in `docs/paper_draft/`.

---

## At-a-Glance Summary

| Day | Theme | Headline deliverable |
|---|---|---|
| 0 | Prep | All accounts/keys ready |
| 1 | Infra | Qdrant + Postgres + Redis running |
| 2 | Ingestion | Clean `chunks.jsonl` |
| 3 | Indexing | Populated Qdrant + citation ground-truth JSONs |
| 4 | Retrieval | Tested hybrid retrieval |
| 5 | Evidence + LLM client | Rate-limited OpenRouter client |
| 6 | Citation verification | Fake-law rejection working |
| 7 | **First demo** | End-to-end Baseline C |
| 8 | Dataset pt.1 | Factoid + reasoning questions |
| 9 | **Dataset complete** | Locked train/eval split, 5 categories |
| 10 | Uncertainty | Calibrated `U` score |
| 11 | Constitutional engine | Full pipeline (pre-fine-tune) |
| 12 | Fine-tuning | Trained QLoRA adapter |
| 13 | Deployment | All 5 baselines, Baseline E live |
| 14 | **Evaluation** | Full results table |
| 15 | **Ship** | Working demo + docs + paper-ready |

**Buffer guidance:** if you slip, the safest days to compress are **8 and 11** (dataset volume can be smaller than target; constitutional prompt-template wording can be refined later). The days you should *never* compress are **9** (locked dataset — everything downstream depends on it) and **14** (your actual results).