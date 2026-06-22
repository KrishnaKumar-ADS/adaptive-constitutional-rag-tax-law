from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel(
    "BAAI/bge-m3",
    use_fp16=True
)

result = model.encode(
    ["What is Section 10 of the Income Tax Act?"],
    return_dense=True,
    return_sparse=True
)

print("Dense:", len(result["dense_vecs"][0]))
print("Sparse:", len(result["lexical_weights"][0]))