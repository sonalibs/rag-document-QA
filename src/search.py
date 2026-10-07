import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from src.chunking import chunk_text
from src.cleaning import clean_text

model= SentenceTransformer("all-MiniLM-L6-v2")

def load_chunks(folder="data/ai_docs"):
    chunks= []
    for path in sorted(Path(folder).glob("*.txt")):
        text= clean_text(path.read_text(encoding="utf-8"))
        for piece in chunk_text(text):
            chunks.append({"text": piece, "source": path.name})
    return chunks

def build_index(chunks):
    texts= [c["text"] for c in chunks]
    return model.encode(texts, normalize_embeddings=True)

def search(question, chunks, vectors, top_k=6, min_score=0.35):
    q = model.encode([question], normalize_embeddings=True)[0]
    scores = vectors @ q
    best = np.argsort(scores)[::-1][:top_k]
    if scores[best[0]] < min_score:
        return []
    return [{"score": float(scores[i]), **chunks[i]} for i in best]