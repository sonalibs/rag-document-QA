from src.search import load_chunks, build_index, search

chunks= load_chunks()
vectors= build_index(chunks)
print("chunks:", len(chunks), "| Vectors shape: ", vectors.shape )

questions = [
    "What is retrieval-augmented generation?",
    "What is a vector database used for?",
    "What is Python?",
    "Who won the 2018 football world cup?",
    "How do I bake sourdough bread?",
]

for q in questions:
    print("Q:", q)
    results = search(q, chunks, vectors)
    if not results:
        print("-> I don't know (nothing relevant found)")
    for r in results:
        print(round(r["score"], 3), r["source"], "|", r["text"][:80])
    print()