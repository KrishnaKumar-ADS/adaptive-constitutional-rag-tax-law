from src.pipeline import run_pipeline

queries = [
    # 1. Factoid (Low Uncertainty)
    "What is the capital gains tax rate under Section 112A?",
    
    # 2. Reasoning (Medium Uncertainty)
    "If a person sells agricultural land, are they subject to capital gains tax?",
    
    # 3. Adversarial / Unanswerable (High Uncertainty)
    "What are the specific penalties under Section 999Z for buying a private jet?"
]

for i, q in enumerate(queries, 1):
    print(f"\n{'='*80}\nQuery {i}: {q}")
    response = run_pipeline(q, top_k=3, log_to_db=False)
    print(f"Decision: {response.verdict}")
    print(f"Confidence: {response.confidence}")
    print(f"Answer:\n{response.answer[:200]}...")
