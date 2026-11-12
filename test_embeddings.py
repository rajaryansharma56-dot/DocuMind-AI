
from sentence_transformers import SentenceTransformer

print("\n========== DOCUMIND AI ==========")

print("\nLoading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [
    "Artificial Intelligence helps computers learn.",
    "Machine learning is a part of artificial intelligence.",
    "Python is a popular programming language."
]

print("\nGenerating embeddings...")

embeddings = model.encode(texts)

print("\nEMBEDDING RESULTS")

for i, embedding in enumerate(embeddings):
    print(f"\nText {i + 1}: {texts[i]}")
    print("Embedding dimensions:", len(embedding))
    print("First 5 values:", embedding[:5])

print("\n========== TEST COMPLETED ==========")
