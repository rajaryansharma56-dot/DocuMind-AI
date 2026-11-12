
import chromadb
from pathlib import Path


class VectorStore:
    """
    Persistent vector database for DocuMind AI.
    """

    def __init__(self, collection_name="documind_documents"):

        self.db_path = Path("data/chroma_db")
        self.db_path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.db_path)
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, chunks, embeddings):

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Chunks and embeddings count must match."
            )

        if not chunks:
            return

        ids = []
        documents = []
        metadatas = []
        vectors = []

        for i, chunk in enumerate(chunks):

            chunk_id = str(chunk.get("chunk_id", i))

            ids.append(chunk_id)
            documents.append(chunk["text"])

            metadatas.append({
                "page_number": int(chunk.get("page_number", 0)),
                "chunk_index": int(chunk.get("chunk_index", i))
            })

            vectors.append(embeddings[i].tolist())

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=vectors
        )

        print(f"Successfully stored {len(chunks)} chunks.")

    def search(self, query_embedding, n_results=3):

        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(n_results, self.collection.count())
        )

        return results

    def get_collection_count(self):

        return self.collection.count()
