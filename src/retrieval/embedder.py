from FlagEmbedding import BGEM3FlagModel

model = None

def get_model():
    global model
    if model is None:
        model = BGEM3FlagModel("BAAI/bge-m3")
    return model

def encode_query(query: str):
    m = get_model()
    return m.encode(
        [query],
        return_dense=True,
        return_sparse=True
    )