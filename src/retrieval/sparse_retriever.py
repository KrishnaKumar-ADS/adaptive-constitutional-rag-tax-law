"""Sparse retrieval using BM25."""
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector

from src.retrieval.embedder import encode_query

client = QdrantClient(
    "localhost",
    port=6333
)

COLLECTION = "tax_law"


def sparse_search(
    query: str,
    top_k: int = 10
):
    encoded = encode_query(query)

    sparse = encoded["lexical_weights"][0]

    indices = list(sparse.keys())
    values = list(sparse.values())

    results = client.search(
        collection_name=COLLECTION,
        query_vector=(
            "sparse",
            SparseVector(
                indices=indices,
                values=values
            )
        ),
        limit=top_k,
        with_payload=True
    )

    return results