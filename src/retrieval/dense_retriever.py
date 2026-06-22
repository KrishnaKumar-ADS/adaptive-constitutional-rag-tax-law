"""Dense vector retrieval using BGE-M3."""
from qdrant_client import QdrantClient
from src.retrieval.embedder import encode_query

client = QdrantClient("localhost", port=6333)

COLLECTION = "tax_law"


def dense_search(
    query: str,
    top_k: int = 10
):
    encoded = encode_query(query)

    dense_vector = encoded["dense_vecs"][0]

    results = client.search(
        collection_name=COLLECTION,
        query_vector=(
            "dense",
            dense_vector
        ),
        limit=top_k,
        with_payload=True
    )

    return results