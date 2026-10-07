from src.search import load_chunks, build_index, search
from src.generate import build_prompt

chunks= load_chunks()
vectors= build_index(chunks)

question= "How does retrieval-augmented generation reduce hallucinations?"
result= search(question, chunks, vectors)
print(build_prompt(question, result))