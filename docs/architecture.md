# Adaptive Constitutional RAG Architecture

## Overview
This document outlines the architecture of the Adaptive Constitutional RAG system, designed to provide highly reliable, verifiable, and legally grounded answers to Indian Tax Law queries.

## Core Components

### 1. Hybrid Retrieval Engine (Day 4-6)
Combines semantic search (BGE-M3) with lexical search (BM25) using Reciprocal Rank Fusion (RRF) on Qdrant. Ensures high recall for both conceptual queries and exact section matches.

### 2. Uncertainty Quantification (Day 8-9)
A calibrated GradientBoostingClassifier predicts the likelihood of the LLM hallucinating based on 4 features:
- `retrieval_confidence`: BM25/Semantic fusion scores
- `evidence_agreement`: NLI entailment between query and context
- `coverage`: Keyword overlap
- `entropy`: Score distribution

### 3. Constitutional Policy Engine (Day 11)
Routes the query through one of three strictness tiers based on the Uncertainty Score ($U$):
- **Tier 1 (Low, $U < 0.2$)**: Standard RAG generation.
- **Tier 2 (Medium, $0.2 \le U < 0.7$)**: LLM prompted to flag ambiguity and cite extensively.
- **Tier 3 (High, $U \ge 0.7$)**: Forced abstention. The system refuses to answer.

### 4. Zero-LLM Verification (Day 10)
A post-generation check using deterministic index lookup and cross-encoder entailment to ensure every section cited actually exists and supports the claim.

### 5. Local Inference (Day 12-13)
A Qwen3-8B model fine-tuned on tax QA pairs using QLoRA/DoRA, quantized to Q4_K_M GGUF, and served locally via Ollama. Replaces the generic OpenRouter model for specialized generation.

## Inference Flow
```mermaid
graph TD
    Q[User Query] --> R[Hybrid Retrieval (Qdrant)]
    R --> E[Evidence Aggregation]
    E --> U[Uncertainty Quantification]
    U -->|Score U| P[Policy Engine]
    
    P -->|U < 0.2| T1[Tier 1: Standard Gen]
    P -->|0.2 <= U < 0.7| T2[Tier 2: Cautious Gen]
    P -->|U >= 0.7| T3[Tier 3: Abstain]
    
    T1 --> G[Local LLM (Qwen-Tax)]
    T2 --> G
    T3 --> A[Return 'Cannot Answer']
    
    G --> V[Zero-LLM Verifier]
    V -->|Valid| O[Return Response]
    V -->|Invalid| H[Log Hallucination]
```
