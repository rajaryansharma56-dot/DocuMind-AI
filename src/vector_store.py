
import hashlib
import math
from pathlib import Path

import chromadb


class VectorStore:

    def __init__(
        self,
        collection_name="documind_documents_v2",
        batch_size=100
    ):

        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("Batch size must be a positive integer.")

        self.project_root = Path(__file__).resolve().parent.parent

        self.db_path = self.project_root / "data" / "chroma_db"

        self.db_path.mkdir(
            parents=True,
            exist_ok=True
        )

        self.batch_size = batch_size
        self.embedding_dimension = None

        self.client = chromadb.PersistentClient(
            path=str(self.db_path)
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    # ==========================================
    # 1. GENERATE DOCUMENT ID
    # ==========================================

    def generate_document_id(self, filename):

        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("Filename must be a non-empty string.")

        normalized_name = filename.strip().lower()

        return hashlib.sha256(
            normalized_name.encode("utf-8")
        ).hexdigest()[:24]

    # ==========================================
    # 2. EXTRACT CHUNK TEXT
    # ==========================================

    def get_chunk_text(self, chunk):

        if isinstance(chunk, str):
            text = chunk

        elif isinstance(chunk, dict):
            text = chunk.get("text")

            if text is None:
                text = chunk.get("content")

        else:
            raise ValueError("Chunk must be a string or dictionary.")

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Chunk text cannot be empty.")

        return text.strip()

    # ==========================================
    # 3. EXTRACT PAGE NUMBER
    # ==========================================

    def get_page_number(self, chunk):

        if not isinstance(chunk, dict):
            return 0

        try:
            page_number = int(chunk.get("page_number", 0))

            if page_number < 0:
                return 0

            return page_number

        except (TypeError, ValueError):
            return 0

    # ==========================================
    # 4. VALIDATE SINGLE EMBEDDING
    # ==========================================

    def validate_embedding(self, embedding, index=0):

        if not isinstance(embedding, (list, tuple)):
            raise ValueError(
                f"Invalid embedding at index {index}."
            )

        if len(embedding) == 0:
            raise ValueError(
                f"Embedding at index {index} cannot be empty."
            )

        validated = []

        for value in embedding:

            if isinstance(value, bool):
                raise ValueError("Embedding values must be numeric.")

            if not isinstance(value, (int, float)):
                raise ValueError("Embedding values must be numeric.")

            if not math.isfinite(value):
                raise ValueError("Embedding contains non-finite values.")

            validated.append(float(value))

        if all(value == 0 for value in validated):
            raise ValueError("Embedding cannot be a zero vector.")

        return validated

    # ==========================================
    # 5. ADD DOCUMENTS
    # ==========================================

    def add_documents(
        self,
        chunks,
        embeddings,
        filename=None,
        document_id=None
    ):

        if not isinstance(chunks, (list, tuple)):
            raise ValueError("Chunks must be a list.")

        if not isinstance(embeddings, (list, tuple)):
            raise ValueError("Embeddings must be a list.")

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks and embeddings must match."
            )

        if len(chunks) == 0:
            return True

        if document_id is None:

            if filename:
                document_id = self.generate_document_id(filename)

            else:
                raise ValueError(
                    "A document ID or filename is required."
                )

        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("Document ID must be a non-empty string.")

        document_id = document_id.strip()

        # Validate all chunk text first
        documents = [
            self.get_chunk_text(chunk)
            for chunk in chunks
        ]

        # Validate all embeddings first
        validated_embeddings = [
            self.validate_embedding(embedding, index)
            for index, embedding in enumerate(embeddings)
        ]

        dimensions = {
            len(embedding)
            for embedding in validated_embeddings
        }

        if len(dimensions) != 1:
            raise ValueError(
                "All embeddings must have the same dimension."
            )

        new_dimension = dimensions.pop()

        if (
            self.embedding_dimension is not None
            and new_dimension != self.embedding_dimension
        ):
            raise ValueError(
                "Embedding dimension does not match the existing database."
            )

        # Prepare metadata and IDs
        ids = []
        metadatas = []

        for index, chunk in enumerate(chunks):

            chunk_id = f"{document_id}_{index}"

            ids.append(chunk_id)

            metadata = {}

            if isinstance(chunk, dict):

                # Preserve additional simple metadata fields
                for key, value in chunk.items():

                    if key in ("text", "content", "metadata"):
                        continue

                    if isinstance(value, (str, int, float, bool)):
                        metadata[key] = value

            # Preserve explicitly supplied nested metadata
            if isinstance(chunk, dict):

                extra_metadata = chunk.get("metadata", {})

                if isinstance(extra_metadata, dict):

                    for key, value in extra_metadata.items():

                        if isinstance(value, (str, int, float, bool)):
                            metadata[key] = value

            metadata["filename"] = filename or "unknown"
            metadata["page_number"] = self.get_page_number(chunk)
            metadata["document_id"] = document_id
            metadata["chunk_index"] = index

            metadatas.append(metadata)

        # Remove old chunks belonging to this document
        old_results = self.collection.get(
            where={"document_id": document_id}
        )

        old_ids = old_results.get("ids", [])

        if old_ids:
            self.collection.delete(ids=old_ids)

        # Insert in batches
        for start in range(0, len(chunks), self.batch_size):

            end = start + self.batch_size

            self.collection.upsert(
                ids=ids[start:end],
                documents=documents[start:end],
                embeddings=validated_embeddings[start:end],
                metadatas=metadatas[start:end]
            )

        self.embedding_dimension = new_dimension

        return True

    # ==========================================
    # 6. SEARCH DOCUMENTS
    # ==========================================

    def search(self, query_embedding, n_results=3):

        if isinstance(n_results, bool) or not isinstance(n_results, int):
            raise ValueError("Result count must be a positive integer.")

        if n_results < 1:
            raise ValueError("Result count must be positive.")

        validated_embedding = self.validate_embedding(
            query_embedding
        )

        total_chunks = self.collection.count()

        if total_chunks == 0:
            return []

        if self.embedding_dimension is not None:

            if len(validated_embedding) != self.embedding_dimension:
                raise ValueError(
                    "Query embedding dimension does not match database."
                )

        n_results = min(n_results, total_chunks)

        results = self.collection.query(
            query_embeddings=[validated_embedding],
            n_results=n_results,
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        ids = results.get("ids", [])
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        distances = results.get("distances", [])

        # ChromaDB normally returns nested lists.
        # This also supports the flat lists used by the test fake.

        def unwrap(values):

            if values and isinstance(values[0], list):
                return values[0]

            return values

        ids = unwrap(ids)
        documents = unwrap(documents)
        metadatas = unwrap(metadatas)
        distances = unwrap(distances)

        formatted_results = []

        for index, content in enumerate(documents):

            formatted_results.append({

                "id": ids[index] if index < len(ids) else None,

                "content": content,

                "metadata": (
                    metadatas[index]
                    if index < len(metadatas)
                    else {}
                ),

                "distance": (
                    distances[index]
                    if index < len(distances)
                    else None
                )

            })

        return formatted_results

    # ==========================================
    # 7. GET ALL CHUNKS
    # ==========================================

    def get_all_chunks(self):

        results = self.collection.get(
            include=[
                "documents",
                "metadatas"
            ]
        )

        ids = results.get("ids", [])
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])

        formatted_results = []

        for index, content in enumerate(documents):

            formatted_results.append({

                "id": ids[index] if index < len(ids) else None,

                "content": content,

                "metadata": (
                    metadatas[index]
                    if index < len(metadatas)
                    else {}
                )

            })

        return formatted_results

    # ==========================================
    # 8. LIST DOCUMENTS
    # ==========================================

    def list_documents(self):

        chunks = self.get_all_chunks()

        documents = {}

        for chunk in chunks:

            metadata = chunk["metadata"]

            document_id = metadata.get("document_id")

            if not document_id:
                continue

            if document_id not in documents:

                documents[document_id] = {

                    "document_id": document_id,

                    "filename": metadata.get(
                        "filename",
                        "unknown"
                    ),

                    "chunks": 0

                }

            documents[document_id]["chunks"] += 1

        return list(documents.values())

    # ==========================================
    # 9. DELETE DOCUMENT
    # ==========================================

    def delete_document(self, document_id):

        if not isinstance(document_id, str) or not document_id.strip():
            return False

        results = self.collection.get(
            where={"document_id": document_id}
        )

        ids = results.get("ids", [])

        if not ids:
            return False

        self.collection.delete(ids=ids)

        return True

    # ==========================================
    # 10. COLLECTION COUNT
    # ==========================================

    def get_collection_count(self):

        return self.collection.count()

    # ==========================================
    # 11. DATABASE STATISTICS
    # ==========================================

    def get_database_stats(self):

        chunks = self.get_all_chunks()
        documents = self.list_documents()

        return {

            "total_chunks": len(chunks),

            "total_documents": len(documents),

            "embedding_dimension": self.embedding_dimension,

            "database_path": str(self.db_path)

        }
