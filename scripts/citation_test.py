"""Debug script: inspect retrieval results for a legal query."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrieval.query_processor import process_query
from src.retrieval.cross_reference import find_cross_references
from src.retrieval.hybrid_retriever import hybrid_search

query = "What is the penalty for failure to comply with Section 206C?"

processed = process_query(query)
print(f"Intent: {processed.intent}")
print(f"Referenced sections: {processed.referenced_sections}")
print(f"Search queries: {processed.search_queries}")

xrefs = find_cross_references(processed.referenced_sections, intent=processed.intent)
print(f"\nCross-references ({len(xrefs)}):")
for x in xrefs[:8]:
    print(f"  - Section {x['section_number']}: {x['title'][:60]}")

print("\nHybrid search results:")
results = hybrid_search(query, top_k=8, processed=processed)
for i, r in enumerate(results):
    sec = r.payload.get("section_number", "N/A")
    title = r.payload.get("title", "N/A")
    print(f"{i+1}. [Score: {r.score:.4f}] Section {sec}: {title}")
