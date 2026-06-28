# Tax Law Q&A Dataset Card

## Overview
This dataset was synthetically generated and curated to fine-tune the Qwen3-8B model for the Adaptive Constitutional RAG system. It focuses on the Indian Income Tax Act 1961 and the Constitution of India.

## Dataset Structure
Total Size: ~3,700 examples (2,945 Train / 737 Eval)

### Categories
1. **Factoid (30%)**: Direct factual queries (e.g., "What is the penalty under Section 271F?").
2. **Reasoning (25%)**: Applying law to a specific scenario.
3. **Multi-hop (20%)**: Requiring synthesis of multiple sections (e.g., connecting Section 10 with Section 80C).
4. **Unanswerable (15%)**: Queries where the context does not contain the answer, training the model to abstain.
5. **Adversarial (10%)**: Tricky queries designed to induce hallucinations (e.g., fake section numbers).

## Formatting
The data is formatted in ChatML to support system prompts with varying strictness tiers.

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a specialized Indian tax law assistant..."
    },
    {
      "role": "user",
      "content": "Context: ... \n\n Question: ..."
    },
    {
      "role": "assistant",
      "content": "According to Section X..."
    }
  ]
}
```

## Generation Methodology
Generated using a teacher model (GPT-4o/Claude 3.5 Sonnet) prompted to extract scenarios from the raw legal text corpus. The dataset was then validated using the `src.citation.verifier` to ensure zero hallucinated citations in the ground truth.
