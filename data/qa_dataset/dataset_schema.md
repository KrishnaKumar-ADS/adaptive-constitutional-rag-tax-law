# QA Dataset Schema

Every JSONL file in this directory must conform to the following schema:

```json
{
  "question": "The actual question text.",
  "answer": "The expected answer text. For unanswerable questions, this should be an explicit abstention.",
  "evidence_citations": ["List", "of", "Section or Article IDs", "needed to answer this"],
  "category": "factoid | reasoning | multi_hop | unanswerable | adversarial",
  "difficulty": "easy | medium | hard"
}
```

## Categories

1. **factoid**: Direct retrieval, explicitly stated in the text (e.g., "What does Section 10 say?").
2. **reasoning**: Requires interpreting a provision (e.g., applying a rule to a simple hypothetical).
3. **multi_hop**: Requires combining at least two different sections/articles.
4. **unanswerable**: Questions about topics not in the corpus (requires abstention).
5. **adversarial**: Questions containing fabricated law (e.g., "Section 999Z") (requires citation rejection).
