"""CLI tool to ask a tax-law question and get a structured response.

Usage:
    python scripts/ask.py "Is agricultural income exempt from tax?"
"""
import sys
import json

from benchmark.baselines.baseline_c_standard_rag import (
    run_baseline_c,
)

from src.db.crud import log_query


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python scripts/ask.py "
            '"Your tax question here"'
        )
        sys.exit(1)

    question = sys.argv[1]

    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}\n")

    response = run_baseline_c(question)

    # Pretty-print the structured response
    response_dict = response.model_dump()

    print(
        json.dumps(
            response_dict,
            indent=2,
            ensure_ascii=False,
        )
    )

    # Day 7: Log to Postgres
    try:
        evidence_ids = [
            ev["citation_id"]
            for ev in response_dict["evidence"]
        ]

        query_id = log_query(
            question=question,
            evidence_ids=evidence_ids,
            decision=response_dict.get("verdict"),
            citations_json={
                "verdict": response_dict.get("verdict"),
                "confidence": response_dict.get("confidence"),
                "evidence": response_dict.get("evidence"),
            },
            raw_response=response_dict.get("answer"),
        )

        if query_id:
            print(
                f"\n✓ Logged to Postgres (query_id={query_id})"
            )
    except Exception as e:
        print(f"\n⚠ DB logging failed: {e}")


if __name__ == "__main__":
    main()