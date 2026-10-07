from src.search import load_chunks, build_index, search

chunks = load_chunks()
vectors = build_index(chunks)

question = "How does retrieval-augmented generation reduce hallucinations?"
phrase = "reduce AI hallucinations"

results = search(question, chunks, vectors, top_k=20, min_score=-1.0)
for rank, r in enumerate(results, start=1):
    marker = "<-- ANSWER" if phrase in r["text"] else ""
    print(rank, round(r["score"], 3), r["source"], marker)