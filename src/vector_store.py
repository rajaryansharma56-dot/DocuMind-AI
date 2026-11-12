
import chromadb
import math

from pathlib import Path
from hashlib import sha256


class VectorStore:

    def __init__(
        self,
        collection_name="documind_documents_v2",
        batch_size=100
    ):

        self.db_path = Path("data/chroma_db")
        self.db_path.mkdir(parents=True, exist_ok=True)

        self.batch_size = batch_size
        self.embedding_dimension = None

        self.client = chromadb.PersistentClient(
            path=str(self.db_path)
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        print("Vector database initialized successfully.")

    # -----------------------------------------------
    # GENERATE DOCUMENT ID
    # -----------------------------------------------

    def generate_document_id(self, filename):

        filename = Path(filename).name.lower()

        return sha256(
            filename.encode("utf-8")
        ).hexdigest()[:24]

    # -----------------------------------------------
    # EXTRACT CHUNK TEXT
    # -----------------------------------------------

    def get_chunk_text(self, chunk):

        if isinstance(chunk, str):
            return chunk.strip()

        if isinstance(chunk, dict):

            text = (
                chunk.get("text")
                or chunk.get("content")
                or chunk.get("chunk_text")
                or ""
            )

            return str(text).strip()

        raise ValueError("Invalid chunk format.")

    # -----------------------------------------------
    # EXTRACT PAGE NUMBER
    # -----------------------------------------------

    def get_page_number(self, chunk):

        if isinstance(chunk, dict):

            page = chunk.get("page_number", 0)

            try:
                return int(page) if page is not None else 0

            except (ValueError, TypeError):
                return 0

        return 0

    # -----------------------------------------------
    # VALIDATE EMBEDDING
    # -----------------------------------------------

    def validate_embedding(self, embedding, index):

        try:

            embedding = [
                float(value) for value in embedding
            ]

        except (TypeError, ValueError):

            raise ValueError(
                f"Embedding {index} contains invalid values."
            )

        if not embedding:
            raise ValueError(
                f"Embedding {index} is empty."
            )

        if not all(math.isfinite(v) for v in embedding):

            raise ValueError(
                f"Embedding {index} contains non-finite values."
            )

        if not any(value != 0 for value in embedding):

            raise ValueError(
                f"Embedding {index} is a zero vector."
            )

        return embedding

    # -----------------------------------------------
    # ADD DOCUMENTS
    # -----------------------------------------------

    def add_documents(
        self,
        chunks,
        embeddings,
        filename="Unknown document",
        document_id=None
    ):

        if len(chunks) != len(embeddings):

            raise ValueError(
                "Number of chunks and embeddings must match."
            )

        if not chunks:
            print("No chunks available.")
            return

        filename = Path(filename).name

        if document_id is None:

            document_id = self.generate_document_id(
                filename
            )

        texts = []
        metadatas = []
        ids = []
        embedding_data = []

        expected_dimension = None

        for index, chunk in enumerate(chunks):

            text = self.get_chunk_text(chunk)

            if not text:

                raise ValueError(
                    f"Chunk {index} is empty."
                )

            embedding = self.validate_embedding(
                embeddings[index],
                index
            )

            if expected_dimension is None:

                expected_dimension = len(embedding)

            elif len(embedding) != expected_dimension:

                raise ValueError(
                    "All embeddings must have identical dimensions."
                )

            page_number = self.get_page_number(chunk)

            texts.append(text)
            embedding_data.append(embedding)

            metadatas.append({
                "document_id": str(document_id),
                "filename": filename,
                "page_number": page_number,
                "chunk_index": index
            })

            ids.append(
                f"{document_id}_chunk_{index}"
            )

        if (
            self.embedding_dimension is not None
            and self.embedding_dimension != expected_dimension
        ):

            raise ValueError(
                "Embedding dimension does not match the "
                "existing database configuration."
            )

        # Find existing chunks before replacement

        existing = self.collection.get(
            where={"document_id": str(document_id)}
        )

        old_ids = existing.get("ids", [])

        # Insert in batches

        for start in range(0, len(texts), self.batch_size):

            end = start + self.batch_size

            self.collection.upsert(
                ids=ids[start:end],
                documents=texts[start:end],
                embeddings=embedding_data[start:end],
                metadatas=metadatas[start:end]
            )

        # Delete obsolete chunks after successful insertion

        obsolete_ids = [
            old_id for old_id in old_ids
            if old_id not in set(ids)
        ]

        if obsolete_ids:

            self.collection.delete(
                ids=obsolete_ids
            )

        self.embedding_dimension = expected_dimension

        print("\nDocument stored successfully!")
        print("Filename:", filename)
        print("Document ID:", document_id)
        print("Chunks stored:", len(chunks))
        print("Embedding dimension:", expected_dimension)
        print("Total database chunks:", self.collection.count())

    # -----------------------------------------------
    # SEMANTIC SEARCH
    # -----------------------------------------------

    def search(self, query_embedding, n_results=3):

        query_embedding = self.validate_embedding(
            query_embedding,
            "query"
        )

        if (
            self.embedding_dimension is not None
            and len(query_embedding) != self.embedding_dimension
        ):

            raise ValueError(
                "Query embedding dimension does not match "
                "stored document embeddings."
            )

        total_chunks = self.collection.count()

        if total_chunks == 0:
            return []

        n_results = int(n_results)

        if n_results < 1:
            raise ValueError(
                "n_results must be positive."
            )

        n_results = min(
            n_results,
            total_chunks
        )

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        formatted_results = []

        result_ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for index, chunk_id in enumerate(result_ids):

            metadata = (
                metadatas[index]
                if index < len(metadatas)
                else None
            ) or {}

            content = (
                documents[index]
                if index < len(documents)
                else ""
            ) or ""

            distance = (
                distances[index]
                if index < len(distances)
                else None
            )

            formatted_results.append({
                "id": chunk_id,
                "content": content,
                "metadata": metadata,
                "distance": distance
            })

        return formatted_results

    # -----------------------------------------------
    # GET ALL CHUNKS
    # -----------------------------------------------

    def get_all_chunks(self):

        results = self.collection.get(
            include=[
                "documents",
                "metadatas"
            ]
        )

        formatted_results = []

        ids = results.get("ids", [])
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])

        for index, chunk_id in enumerate(ids):

            content = (
                documents[index]
                if index < len(documents)
                else ""
            ) or ""

            metadata = (
                metadatas[index]
                if index < len(metadatas)
                else None
            ) or {}

            if not content.strip():
                continue

            formatted_results.append({
                "id": chunk_id,
                "content": content,
                "metadata": metadata
            })

        return formatted_results

    # -----------------------------------------------
    # LIST DOCUMENTS
    # -----------------------------------------------

    def list_documents(self):

        results = self.collection.get(
            include=["metadatas"]
        )

        documents = {}

        ids = results.get("ids", [])
        metadatas = results.get("metadatas", [])

        for index, chunk_id in enumerate(ids):

            metadata = (
                metadatas[index]
                if index < len(metadatas)
                else None
            ) or {}

            document_id = metadata.get(
                "document_id",
                "legacy_" + chunk_id
            )

            filename = metadata.get(
                "filename",
                "Unknown document"
            )

            if document_id not in documents:

                documents[document_id] = {
                    "document_id": document_id,
                    "filename": filename,
                    "chunks": 0
                }

            documents[document_id]["chunks"] += 1

        return list(documents.values())

    # -----------------------------------------------
    # DELETE DOCUMENT
    # -----------------------------------------------

    def delete_document(self, document_id):

        existing = self.collection.get(
            where={"document_id": str(document_id)}
        )

        ids = existing.get("ids", [])

        if not ids:

            print("Document not found.")
            return False

        self.collection.delete(ids=ids)

        print(
            f"Deleted {len(ids)} chunks successfully."
        )

        return True

    # -----------------------------------------------
    # DATABASE COUNT
    # -----------------------------------------------

    def get_collection_count(self):

        return self.collection.count()
