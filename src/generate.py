def build_prompt(question, results):
    context = ""
    for i, r in enumerate(results, start=1):
        context += f"[{i}] (source: {r['source']})\n{r['text']}\n\n"
    return (
        "Answer the question using ONLY the context below. "
        "Cite the sources you used like [1] or [2]. "
        "If the context does not contain the answer, say you don't know.\n\n"
        f"Context:\n{context}"
        f"Question: {question}\nAnswer:"
    )