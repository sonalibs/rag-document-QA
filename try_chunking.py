from src.chunking import chunk_text

text = open("data/ai_docs/retrieval-augmented_generation.txt", encoding="utf-8").read()
chunks = chunk_text(text)

print("Total characters:", len(text))
print("Number of chunks:", len(chunks))
print("--- Chunk 0 ---")
print(chunks[0])