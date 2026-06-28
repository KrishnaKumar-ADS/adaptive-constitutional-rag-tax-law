import asyncio
from src.retrieval.hybrid_retriever import hybrid_search
from src.evidence.evidence_aggregator import aggregate_evidence
from src.llm.inference_local import generate_local
from src.llm.inference_groq import reformat_and_score

async def main():
    q = "What is the penalty for failure to comply with Section 206C?"
    results = hybrid_search(q, 8)
    evidence_set = aggregate_evidence(results)
    
    flat_evidence = "\n\n".join(
        f"[Section {e.section_number or e.article_number or 'Unknown'}] {e.text}"
        for e in evidence_set.evidences
    )
    
    print("--- FETCHED EVIDENCE ---")
    print(flat_evidence)
    print("------------------------")
    
    resp = await generate_local(q, flat_evidence, "medium")
    print("\n--- RAW QWEN OUTPUT ---")
    print(resp["text"])
    print("-----------------------\n")
    
    groq = await reformat_and_score(resp["text"], q, flat_evidence)
    print("GROQ SCORE:", groq["quality_score"])
    print("GROQ ISSUES:", groq["issues"])

asyncio.run(main())
