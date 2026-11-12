
from sentence_transformers import SentenceTransformer, util

print("\n========== DOCUMIND AI ==========")

# Step 1: Load the embedding model
print("\nLoading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")


# Step 2: Sample document chunks
document_chunks = [
    "Artificial Intelligence enables computers to perform tasks that normally require human intelligence.",

    "Machine learning is a branch of artificial intelligence that allows computers to learn from data.",

    "Python is a popular programming language used in artificial intelligence and data science.",

    "Deep learning uses neural networks with multiple layers to learn complex patterns.",

    "Natural Language Processing helps computers understand and process human language."
]


# Step 3: Generate embeddings for document chunks
print("\nGenerating document embeddings...")

chunk_embeddings = model.encode(
    document_chunks,
    convert_to_tensor=True
)


# Step 4: Get user search query
query = input("\nEnter your search question: ")

query_embedding = model.encode(
    query,
    convert_to_tensor=True
)


# Step 5: Calculate semantic similarity
similarity_scores = util.cos_sim(
    query_embedding,
    chunk_embeddings
)[0]


# Step 6: Rank the most relevant chunks
results = []

for i, score in enumerate(similarity_scores):
    results.append({
        "text": document_chunks[i],
        "score": float(score)
    })

results.sort(key=lambda x: x["score"], reverse=True)


# Step 7: Display search results
print("\n========== SEARCH RESULTS ==========")

for i, result in enumerate(results[:3], start=1):

    print(f"\nResult {i}")
    print("Text:", result["text"])
    print("Similarity:", round(result["score"], 4))


print("\n========== SEARCH COMPLETED ==========")
