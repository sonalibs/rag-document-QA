from src.search import load_chunks, build_index
from src.generate import answer

chunks = load_chunks()
vectors = build_index(chunks)

questions = [
    "What is a vector database used for?",
    "What is the attention mechanism?",
    "Who won the 2018 football world cup?",
]

for q in questions:
    reply, sources = answer(q, chunks, vectors)
    print("Q:", q)
    print(reply)
    print("Sources:", ", ".join(sources) if sources else "none")
    print()