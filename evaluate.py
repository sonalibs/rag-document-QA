from src.search import load_chunks, build_index, search

chunks= load_chunks()
vectors= build_index(chunks)

in_scope = [
    ("What is chain-of-thought prompting?", "prompt_engineering.txt"),
    ("What is prompt injection?", "prompt_engineering.txt"),
    ("How does flash attention reduce memory usage?", "attention_machine_learning.txt"),
    ("What is the attention mechanism?", "attention_machine_learning.txt"),
    ("What does positional encoding do in a transformer?", "transformer_deep_learning.txt"),
    ("What is GloVe?", "word_embedding.txt"),
    ("How does retrieval-augmented generation reduce hallucinations?", "retrieval-augmented_generation.txt"),
    ("What is hybrid search in RAG?", "retrieval-augmented_generation.txt"),
    ("What is a vector database used for?", "vector_database.txt"),
    ("What is reinforcement learning from human feedback?", "large_language_model.txt"),
]

out_of_scope = [
    "What is Python?",
    "Who is the CEO of OpenAI?",
    "How does a car engine work?",
    "What is the weather in Bengaluru today?",
]

hits = 0
answered = 0
for question, expected in in_scope:
    top = search(question, chunks, vectors, min_score=-1.0)
    sources= [r["source"] for r in top]
    kept= search(question, chunks, vectors)
    if expected in sources:
        hits += 1
    if kept:
        answered += 1 
    status = "HIT " if expected in sources else "MISS"
    print(status, round(top[0]["score"], 3), question, "->", sources)


refused= 0
print()
for question in out_of_scope:
    kept = search(question, chunks, vectors)
    if not kept:
        refused += 1
    print("REFUSED " if not kept else "ANSWERED", question)

print()
print(f"Retrival hit on top 3: {hits}/{len(in_scope)}")
print(f" Not wrongly refused: {answered}/{len(in_scope)}")
print(f"Correctly refused: {refused}/{len(out_of_scope)}")

in_scores = []
for question, expected in in_scope:
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