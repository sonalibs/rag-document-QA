import numpy as np
from sentence_transformers import SentenceTransformer

model= SentenceTransformer("all-MiniLM-L6-v2")
sentences= [
    "How do I rest my password",
    "I forget my login credentials",
    "The weather is sunny today"
    ]

vectors= model.encode(sentences)

print("shape:", vectors.shape)
print("First 5 numbers of sentence 0:", vectors[0][:5])

def cosine(a, b):
    return float(np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b)))


print("password vs credentials:", cosine(vectors[0], vectors[1]))
print("password vs weather:", cosine(vectors[0], vectors[2]))