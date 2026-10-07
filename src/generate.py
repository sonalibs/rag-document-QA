import requests
from src.search import search

def build_prompt(question, results):
    context = ""
    for i, r in enumerate(results, start=1):
        context += f"[{i}] (source: {r['source']})\n{r['text']}\n\n"
    return (
        "Answer the question using ONLY the context below. "
        "Do not add facts that are not in the context. "
        "Keep the answer to 2-4 sentences, and put the source number like [1] right after each claim. "
        "If the context does not contain the answer, say you don't know.\n\n"
        f"Context:\n{context}"
        f"Question: {question}\nAnswer:"
    )

def ask_llm(prompt):
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "llama3.2",
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0},
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]

def answer(question, chunks, vectors):
    results = search(question, chunks, vectors)
    if not results:
        return "I don't know - nothing relevant found in the documents.", []
    reply = ask_llm(build_prompt(question, results))
    sources = sorted({r["source"] for r in results})
    return reply, sources