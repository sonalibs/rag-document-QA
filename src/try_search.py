from src.search import load_chunks, build_index, search

chunks= load_chunks()
vectors= build_index(chunks)
print("chunks:", len(chunks), "| Vectors shape: ", vectors.shape )

for r in search("What is the attention mechanism", chunks,vectors):
    print(round(r["score"], 3), r ["source"])
    print(r["text"][:200])
    print()