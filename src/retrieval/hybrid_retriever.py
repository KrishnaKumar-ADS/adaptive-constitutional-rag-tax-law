"""Hybrid dense+sparse retrieval with RRF fusion."""
from qdrant_client import QdrantClient
from qdrant_client import models

from src.retrieval.embedder import encode_query

client = QdrantClient(
    "localhost",
    port=6333
)

COLLECTION = "tax_law"


def hybrid_search(
    query: str,
    top_k: int = 8
):
    encoded = encode_query(query)

    dense_vector = encoded["dense_vecs"][0]

    sparse = encoded["lexical_weights"][0]

    sparse_vector = models.SparseVector(
        indices=list(sparse.keys()),
        values=list(sparse.values())
    )

    results = client.query_points(
        collection_name=COLLECTION,
        prefetch=[
            models.Prefetch(
                query=dense_vector,
                using="dense",
                limit=50
            ),
            models.Prefetch(
                query=sparse_vector,
                using="sparse",
                limit=50
            )
        ],
        query=models.FusionQuery(
            fusion=models.Fusion.RRF
        ),
        limit=top_k,
        with_payload=True
    )

    return results.points