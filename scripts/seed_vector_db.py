"""Populate Qdrant vector database."""
import json
from tqdm import tqdm

from FlagEmbedding import BGEM3FlagModel

from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    SparseVector,
)

COLLECTION_NAME = "tax_law"
CHUNKS_FILE = "data/processed/chunks.jsonl"

BATCH_SIZE = 16

import os

client = QdrantClient(
    url=os.environ.get("QDRANT_HOST", "http://localhost:6333")
)

print("Loading BGE-M3...")

model = BGEM3FlagModel(
    "BAAI/bge-m3"
)

print("Loading chunks...")

chunks = []

with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line))

print(f"Loaded {len(chunks)} chunks")

for start_idx in tqdm(
    range(0, len(chunks), BATCH_SIZE),
    desc="Embedding"
):

    batch = chunks[start_idx:start_idx + BATCH_SIZE]

    texts = [
        chunk["text"]
        for chunk in batch
    ]

    embeddings = model.encode(
        texts,
        return_dense=True,
        return_sparse=True
    )

    points = []

    for i, chunk in enumerate(batch):

        dense_vec = embeddings["dense_vecs"][i]

        lexical_weights = embeddings["lexical_weights"][i]

        sparse_indices = [
            int(k)
            for k in lexical_weights.keys()
        ]

        sparse_values = [
            float(v)
            for v in lexical_weights.values()
        ]

        payload = {
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            **chunk["metadata"]
        }

        points.append(
            PointStruct(
                id=start_idx + i,

                vector={
                    "dense": dense_vec,
                    "sparse": SparseVector(
                        indices=sparse_indices,
                        values=sparse_values
                    )
                },

                payload=payload
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True
    )

print("\nVector DB population complete")
print(f"Inserted {len(chunks)} chunks")