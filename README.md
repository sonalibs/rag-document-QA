# Document Q&A (RAG) built from scratch

A small retrieval-augmented generation (RAG) system that answers questions from a set of documents, cites which files the answer came from, and says "I don't know" when the documents don't contain the answer.

It is built step by step in plain Python (no LangChain or vector-database framework) so every stage can be inspected, measured and explained: cleaning, chunking, embedding, search, relevance cut-off, prompting and generation. A small evaluation script measures retrieval quality and the refusal behaviour.

## How it works

```
question
   |
   v
embed question (all-MiniLM-L6-v2, 384 dims)
   |
   v
cosine similarity vs. every chunk  --->  take the 6 best chunks
   |
   v
is the BEST score >= 0.35 ?
   | no                          | yes
   v                             v
"I don't know"          build prompt: numbered chunks + rules
(LLM is not called)                |
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
| Chunk | `src/chunking.py` | 400-character chunks, 100-character overlap, cut on word boundaries |
| Search | `src/search.py` | Embeds chunks and the question, returns the top 6, refuses if the best score is below 0.35 |
| Generate | `src/generate.py` | Builds the prompt with numbered sources, calls Ollama, returns answer + sources |
| Evaluate | `evaluate.py` | Measures retrieval and refusal on a small hand-written question set |
| Debug | `find_rank.py` | Shows where the chunk containing a given phrase ranks for a question |

## Design decisions (and why)

- **Cleaning before chunking.** The raw Wikipedia text contained hundreds of lines with a single math symbol each. They were embedded as if they were content and polluted the top search results. Removing lines shorter than 6 words (except headings) removed this noise. Known trade-off: a few legitimately short list items are dropped.
- **Chunks cut on word boundaries.** Fixed-size cuts started chunks mid-word. The chunker now backs up to the nearest space and no longer produces duplicate tail chunks.
- **400-character chunks, top 6.** With 800-character chunks, two test questions never reached the top 10, because the sentence that answers them shares a chunk with unrelated text. Smaller chunks ranked those answers much higher. Giving the model 6 chunks of 400 characters is the same amount of text as 3 chunks of 800, so the comparison is at equal context size.
- **Refusal in two layers.** (1) If even the best chunk scores below 0.35, the LLM is not called and the system answers "I don't know". (2) The prompt tells the model to use only the numbered context and to say it does not know otherwise. In testing, layer 2 caught two out-of-scope questions that passed layer 1.
- **The cut-off checks the best score only.** Applying it to every chunk would silently drop lower-ranked chunks that carry the answer.
- **The cut-off was recalibrated after the chunk size changed.** Smaller chunks produce higher scores for every question, so the best cut-off moved from about 0.30 to 0.35. It was chosen from a measured sweep, leaning low because wrongly refusing is final while a false accept can still be caught by layer 2.
- **Grounded prompt, temperature 0.** The model is told not to add outside facts, to keep answers short and to put a citation like [1] after each claim.

## Evaluation

`evaluate.py` runs 10 answerable and 4 out-of-scope questions. For each answerable question I picked a short evidence phrase that must appear in a retrieved chunk, so the check is about the right *passage*, not just the right file. (My first version only checked the file name, which overstated quality.)

Current results with the settings above:

| Metric | Result |
|---|---|
| Right passage in the top 6 chunks | 9 / 10 |
| Answerable questions not wrongly refused | 9 / 10 |
| Out-of-scope questions refused by the score cut-off | 3 / 4 |

The one out-of-scope question that passes the cut-off ("Who is the CEO of OpenAI?", score 0.45, because the documents discuss OpenAI models) is refused by the LLM with "I don't know". The script also prints a sweep of cut-offs from 0.25 to 0.50 so the trade-off is visible.

## Run it

Requires Python 3.10+ and [Ollama](https://ollama.com) with the `llama3.2` model.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

ollama pull llama3.2        # once
# make sure the Ollama app / `ollama serve` is running

python3 fetch_docs.py       # optional: the documents are already in data/ai_docs/
python3 try_answer.py       # asks sample questions end to end
python3 evaluate.py         # retrieval + refusal evaluation
```

The `try_*.py` scripts are small demos of each stage (chunking, embeddings, search, prompt, full answer).

## Limitations

- **Rare names are hard for embeddings.** "What is GloVe?" misses (rank 10, outside the top 6): the documents only mention GloVe inside a list of software names. Keyword search, or hybrid search combining keyword and vector search, would address this.
- **A small local model gives one-sided answers.** For "How does RAG reduce hallucinations?" the right evidence is retrieved (rank 3) but the 3B model quotes only the caveat that RAG does not prevent hallucinations. A prompt change did not fix it. I expect a larger model to do better but did not test one.
- **The evaluation set is small and hand-made.** Ten answerable and four out-of-scope questions; the cut-off was tuned on the same questions, so the numbers are indicative, not a guarantee. A larger held-out set is needed.
- **Only retrieval and refusal are measured automatically.** Answer quality (faithfulness and completeness) was checked by reading a few answers.
- **Sources are listed even when the model refuses**, because the list shows what was retrieved, not what was used.
- Search is a brute-force NumPy dot product, which is fine for about 300 chunks. A larger corpus would need an index such as FAISS or a vector database.

## Next steps

- Hybrid search (keyword + vector) and a cross-encoder re-ranker, measured against the same evaluation set.
- A larger, held-out evaluation set and an answer-level faithfulness check.
- A FastAPI endpoint around `answer()`.

## Data

The documents are Wikipedia articles, licensed CC BY-SA 4.0.
