from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel(
    "BAAI/bge-m3",
    use_fp16=True
)

def encode_query(query: str):
    return model.encode(
        [query],
        return_dense=True,
        return_sparse=True
    )