from src.search import load_chunks, build_index
from src.generate import answer

chunks = load_chunks()
vectors = build_index(chunks)

questions = [
    "Who is the CEO of OpenAI?",
    "How does retrieval-augmented generation reduce hallucinations?",
    "What is Python?",
]

for q in questions:
    reply, sources = answer(q, chunks, vectors)
    print("Q:", q)
    print(reply)
    print("Sources:", ", ".join(sources) if sources else "none")
    print()