from src.search import load_chunks, build_index, search
from src.generate import build_prompt, ask_llm

chunks= load_chunks()
vectors= build_index(chunks)

question = "What is a vector database used for?"
results= search(question, chunks, vectors)
prompt= build_prompt(question, results)

print(ask_llm(prompt))