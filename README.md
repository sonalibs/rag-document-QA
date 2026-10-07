# Document Q&A (RAG) built from scratch

A small retrieval-augmented generation (RAG) system that answers questions from a set of documents, cites which files the answer came from, and says "I don't know" when the documents don't contain the answer.

It is built step by step in plain Python (no LangChain / vector-DB framework) so every stage can be inspected and explained: cleaning, chunking, embedding, search, relevance cut-off, prompting and generation.

## How it works

```
question
   |
   v
embed question (all-MiniLM-L6-v2, 384 dims)
   |
   v
cosine similarity vs. every chunk  --->  keep top 3 with score >= 0.35
   |                                          |
   | (nothing passes)                          v
   v                                   build prompt: numbered chunks + rules
"I don't know"                                |
                                              v
                                  local LLM (Ollama, llama3.2, temperature 0)
                                              |
                                              v
                                  answer with [1], [2] citations + source files
```

| Step | File | What it does |
|---|---|---|
| Fetch | `fetch_docs.py` | Downloads 7 Wikipedia articles on AI topics into `data/ai_docs/` |
| Clean | `src/cleaning.py` | Drops math-formula debris (one symbol per line) and keeps headings |
| Chunk | `src/chunking.py` | 800-character chunks, 100-character overlap, cut on word boundaries |
| Search | `src/search.py` | Embeds chunks and the question, ranks by cosine similarity, applies the score cut-off |
| Generate | `src/generate.py` | Builds the prompt with numbered sources, calls Ollama, returns answer + sources |

## Design decisions (and why)

- **Cleaning before chunking.** The raw Wikipedia text contained hundreds of lines with a single math symbol each. They were embedded as if they were content and polluted the top search results. Removing lines shorter than 6 words (except headings) cut this noise. Known trade-off: a few legitimately short list items are dropped.
- **Chunks cut on word boundaries.** Fixed-size cuts started chunks mid-word ("eural networks..."). The chunker now backs up to the nearest space and no longer produces duplicate tail chunks.
- **Relevance threshold of 0.35.** Vector search always returns the nearest chunks, even for an unrelated question. I measured top scores for sample questions: answerable questions scored 0.548 or higher, an out-of-scope question ("What is Python?") scored 0.296, and clearly unrelated ones about 0.1. The cut-off sits in the gap. If no chunk passes, the LLM is not called at all.
- **Grounded prompt.** The model is told to use only the numbered context, not to add outside facts, to keep answers short, and to place a citation like [1] after each claim. A second instruction tells it to say it does not know if the context lacks the answer, as a backup to the score threshold.
- **Temperature 0** for repeatable, factual answers.

## Run it

Requires Python 3.10+ and [Ollama](https://ollama.com) with the `llama3.2` model.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

ollama pull llama3.2        # once
# make sure the Ollama app / `ollama serve` is running

python3 fetch_docs.py       # optional: the documents are already in data/ai_docs/
python3 try_answer.py       # asks three sample questions
```

The `try_*.py` scripts are small demos of each stage (chunking, embeddings, search, prompt, full answer).

## Limitations

- Search is a brute-force NumPy dot product, which is fine for about 140 chunks. A larger corpus would need an index such as FAISS or a vector database.
- Ranking inside the top results is imprecise: for "What is retrieval-augmented generation?" a chunk from the prompt-engineering article outscored the dedicated RAG article. Passing several chunks to the model compensates, but it is a real weakness of this small embedding model.
- The 0.35 threshold was calibrated on a handful of questions, not a proper test set.
- A 3B local model can still add small details that are not in the retrieved text, and it cites per answer rather than guaranteeing every claim is supported.
- No formal evaluation yet.

## Next steps

- A small evaluation set (about 10 questions with the expected source document) to measure retrieval hit rate and tune the threshold.
- A faithfulness check on generated answers.
- A FastAPI endpoint around `answer()`.

## Data

The documents are Wikipedia articles, licensed CC BY-SA 4.0.
