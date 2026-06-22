from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
)

COLLECTION_NAME = "tax_law"

client = QdrantClient(
    url="http://localhost:6333"
)

# Delete old collection if it exists
try:
    client.delete_collection(COLLECTION_NAME)
    print(f"Deleted existing collection: {COLLECTION_NAME}")
except Exception:
    pass

# Create collection
client.create_collection(
    collection_name=COLLECTION_NAME,

    vectors_config={
        "dense": VectorParams(
            size=1024,              # BGE-M3 dense dimension
            distance=Distance.COSINE
        )
    },

    sparse_vectors_config={
        "sparse": SparseVectorParams()
    }
)

print(f"\n✓ Collection '{COLLECTION_NAME}' created successfully")

info = client.get_collection(COLLECTION_NAME)

print("\nCollection Details:")
print(f"Name: {COLLECTION_NAME}")
print("Dense Vector Size: 1024")
print("Distance Metric: COSINE")
print("Sparse Vector: Enabled")
print(f"Status: {info.status}")