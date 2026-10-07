from src.search import load_chunks, build_index, search

chunks= load_chunks()
vectors= build_index(chunks)

in_scope = [
    ("What is chain-of-thought prompting?", "chain-of-thought prompting"),
    ("What is prompt injection?", "Prompt injection is a type of cybersecurity attack"),
    ("How does flash attention reduce memory usage?", "Flash attention is an implementation"),
    ("What is the attention mechanism?", "attention is a method that determines"),
    ("What does positional encoding do in a transformer?", "positional encodings"),
    ("What is GloVe?", "Stanford University's GloVe"),
    ("How does retrieval-augmented generation reduce hallucinations?", "reduce AI hallucinations"),
    ("What is hybrid search in RAG?", "Hybrid search"),
    ("What is a vector database used for?", "A vector database, vector store or vector search engine"),
    ("What is reinforcement learning from human feedback?", "reinforcement learning from human feedback"),
]

out_of_scope = [
    "What is Python?",
    "Who is the CEO of OpenAI?",
    "How does a car engine work?",
    "What is the weather in Bengaluru today?",
]

hits = 0
answered = 0
for question, evidence in in_scope:
    top = search(question, chunks, vectors, min_score=-1.0)
    kept = search(question, chunks, vectors)
    hit = any(evidence.lower() in r["text"].lower() for r in top)
    if hit:
        hits += 1
    if kept:
        answered += 1
    status = "HIT " if hit else "MISS"
    print(status, round(top[0]["score"], 3), question)


refused= 0
print()
for question in out_of_scope:
    kept = search(question, chunks, vectors)
    if not kept:
        refused += 1
    print("REFUSED " if not kept else "ANSWERED", question)

print()
print(f"Right passage in top 3: {hits}/{len(in_scope)}")
print(f"Not wrongly refused: {answered}/{len(in_scope)}")
print(f"Correctly refused: {refused}/{len(out_of_scope)}")

in_scores = []
for question, evidence in in_scope:
    top = search(question, chunks, vectors, min_score=-1.0)
    in_scores.append(top[0]["score"])

out_scores = []
for question in out_of_scope:
    top = search(question, chunks, vectors, min_score=-1.0)
    out_scores.append(top[0]["score"])

print()
print("out-of-scope top scores:", [round(s, 3) for s in out_scores])
print()
print("threshold | in-scope answered | out-of-scope refused")
for t in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    answered_n = sum(1 for s in in_scores if s >= t)
    refused_n = sum(1 for s in out_scores if s < t)
    print(f"  {t:.2f}    |  {answered_n}/{len(in_scores)}               |  {refused_n}/{len(out_scores)}")