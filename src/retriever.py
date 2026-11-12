
import re
import hashlib
import math
import json

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from src.embedding_engine import generate_embedding
from src.vector_store import VectorStore


class Retriever:

    def __init__(self):

        self.vector_store = VectorStore()

        # Cross-encoder configuration
        self.reranker = None

        self.reranker_model_name = (
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

        # BM25 cache
        self.bm25 = None
        self.index_signature = None
        self.indexed_documents = []

        print("Advanced Hybrid Retriever initialized.")

    # --------------------------------------------------
    # TOKENIZATION
    # --------------------------------------------------

    def tokenize(self, text):

        if not isinstance(text, str):
            return []

        return re.findall(
            r"[a-zA-Z0-9]+(?:[.+#-][a-zA-Z0-9+#-]*)*",
            text.lower()
        )

    # --------------------------------------------------
    # DOCUMENT SIGNATURE
    # --------------------------------------------------

    def get_document_signature(self, documents):

        document_data = [
            {
                "id": str(doc.get("id", "")),
                "content": str(doc.get("content", ""))
            }
            for doc in documents
        ]

        serialized = json.dumps(
            document_data,
            sort_keys=True,
            ensure_ascii=False
        )

        return hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest()

    # --------------------------------------------------
    # RESET BM25 CACHE
    # --------------------------------------------------

    def reset_bm25(self):

        self.bm25 = None
        self.index_signature = None
        self.indexed_documents = []

    # --------------------------------------------------
    # BUILD BM25 INDEX
    # --------------------------------------------------

    def build_bm25_index(self, documents):

        if not documents:

            self.reset_bm25()
            return None

        signature = self.get_document_signature(documents)

        if (
            self.bm25 is not None
            and signature == self.index_signature
        ):
            return self.bm25

        print("Building BM25 keyword index...")

        tokenized_documents = [
            self.tokenize(doc.get("content", ""))
            for doc in documents
        ]

        if not any(tokenized_documents):

            self.reset_bm25()
            return None

        self.bm25 = BM25Okapi(tokenized_documents)

        self.index_signature = signature
        self.indexed_documents = documents.copy()

        print("BM25 index built successfully.")

        return self.bm25

    # --------------------------------------------------
    # SEMANTIC SEARCH
    # --------------------------------------------------

    def semantic_search(self, query, candidate_k):

        query_embedding = generate_embedding(query)

        if query_embedding is None:
            return []

        results = self.vector_store.search(
            query_embedding=query_embedding,
            n_results=candidate_k
        )

        return results if results else []

    # --------------------------------------------------
    # KEYWORD SEARCH
    # --------------------------------------------------

    def keyword_search(
        self,
        query,
        documents,
        bm25,
        candidate_k
    ):

        if bm25 is None:
            return []

        query_tokens = self.tokenize(query)

        if not query_tokens:
            return []

        scores = bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True
        )

        results = []

        for index in ranked_indices:

            if len(results) >= candidate_k:
                break

            score = float(scores[index])

            if not math.isfinite(score) or score <= 0:
                continue

            result = documents[index].copy()

            result["bm25_score"] = score

            results.append(result)

        return results

    # --------------------------------------------------
    # RECIPROCAL RANK FUSION
    # --------------------------------------------------

    def reciprocal_rank_fusion(
        self,
        semantic_results,
        keyword_results,
        k=60
    ):

        fused_results = {}

        search_groups = [
            semantic_results,
            keyword_results
        ]

        for search_results in search_groups:

            seen_ids = set()

            rank = 0

            for result in search_results:

                if not isinstance(result, dict):
                    continue

                chunk_id = result.get("id")

                if chunk_id is None:
                    continue

                chunk_id = str(chunk_id)

                if not chunk_id.strip() or chunk_id in seen_ids:
                    continue

                seen_ids.add(chunk_id)

                rank += 1

                if chunk_id not in fused_results:

                    fused_results[chunk_id] = result.copy()

                    fused_results[chunk_id]["id"] = chunk_id

                    fused_results[chunk_id]["rrf_score"] = 0.0

                fused_results[chunk_id]["rrf_score"] += (
                    1.0 / (k + rank)
                )

        return sorted(
            fused_results.values(),
            key=lambda item: item["rrf_score"],
            reverse=True
        )

    # --------------------------------------------------
    # LOAD CROSS-ENCODER
    # --------------------------------------------------

    def load_reranker(self):

        if self.reranker is None:

            print("\nLoading Cross-Encoder model...")

            self.reranker = CrossEncoder(
                self.reranker_model_name
            )

            print("Cross-Encoder loaded successfully.")

        return self.reranker

    # --------------------------------------------------
    # RERANK RESULTS
    # --------------------------------------------------

    def rerank_results(self, query, results):

        if not results:
            return []

        reranker = self.load_reranker()

        pairs = [
            [
                query,
                result.get("content", "")
            ]
            for result in results
        ]

        scores = reranker.predict(
            pairs,
            show_progress_bar=False
        )

        if len(scores) != len(results):
            raise ValueError(
                "Reranker returned an invalid score count."
            )

        reranked_results = []

        for index, score in enumerate(scores):

            score = float(score)

            if not math.isfinite(score):
                score = -1e9

            result = results[index].copy()

            result["rerank_score"] = score

            reranked_results.append(result)

        return sorted(
            reranked_results,
            key=lambda item: item["rerank_score"],
            reverse=True
        )

    # --------------------------------------------------
    # NORMALIZE FINAL RESULTS
    # --------------------------------------------------

    def normalize_results(self, results):

        final_results = []

        for result in results:

            metadata = result.get("metadata") or {}

            if not isinstance(metadata, dict):
                metadata = {}

            content = result.get("content", "")

            if not isinstance(content, str) or not content.strip():
                continue

            final_results.append({

                "id": result.get("id"),

                "content": content.strip(),

                "metadata": metadata,

                "filename": metadata.get(
                    "filename",
                    result.get("filename", "Unknown document")
                ),

                "page_number": metadata.get(
                    "page_number",
                    result.get("page_number", "Unknown")
                ),

                "chunk_index": metadata.get(
                    "chunk_index",
                    result.get("chunk_index", -1)
                ),

                "distance": result.get("distance"),

                "bm25_score": result.get("bm25_score"),

                "rrf_score": result.get("rrf_score"),

                "rerank_score": result.get("rerank_score")

            })

        return final_results

    # --------------------------------------------------
    # MAIN HYBRID RETRIEVAL
    # --------------------------------------------------

    def retrieve(self, query, top_k=3):

        if not isinstance(query, str) or not query.strip():
            return []

        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or top_k < 1
        ):
            raise ValueError(
                "top_k must be a positive integer."
            )

        query = query.strip()

        print("\n========== ADVANCED HYBRID RETRIEVAL ==========\n")

        # STEP 1: Load stored chunks

        documents = self.vector_store.get_all_chunks()

        if not documents:

            self.reset_bm25()

            print("No documents found.")

            return []

        documents = [
            doc for doc in documents
            if isinstance(doc, dict)
            and isinstance(doc.get("content"), str)
            and doc["content"].strip()
            and doc.get("id") is not None
        ]

        if not documents:
            self.reset_bm25()
            return []

        print("Total indexed chunks:", len(documents))

        candidate_k = min(
            len(documents),
            max(30, top_k * 10)
        )

        # STEP 2: Semantic search

        print("\n[1] Semantic Search...")

        semantic_results = self.semantic_search(
            query,
            candidate_k
        )

        print("Semantic candidates:", len(semantic_results))

        # STEP 3: BM25 search

        print("\n[2] BM25 Keyword Search...")

        bm25 = self.build_bm25_index(documents)

        keyword_results = self.keyword_search(
            query,
            documents,
            bm25,
            candidate_k
        )

        print("Keyword candidates:", len(keyword_results))

        # STEP 4: Reciprocal Rank Fusion

        print("\n[3] Reciprocal Rank Fusion...")

        fused_results = self.reciprocal_rank_fusion(
            semantic_results,
            keyword_results
        )

        print("Unique fused candidates:", len(fused_results))

        if not fused_results:

            print("No relevant results found.")

            return []

        # STEP 5: Cross-encoder reranking

        print("\n[4] Cross-Encoder Reranking...")

        try:

            reranked_results = self.rerank_results(
                query,
                fused_results
            )

        except Exception as error:

            print("Reranking failed:", error)

            print("Using RRF ranking as fallback.")

            reranked_results = fused_results

        # STEP 6: Final selection

        final_results = self.normalize_results(
            reranked_results[:top_k]
        )

        print("\n========== RETRIEVAL COMPLETE ==========")

        print("Final selected chunks:", len(final_results))

        return final_results
