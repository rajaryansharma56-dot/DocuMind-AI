
import pytest

from unittest.mock import patch

from src.vector_store import VectorStore


# ==========================================
# FAKE CHROMADB COLLECTION
# ==========================================

class FakeCollection:

    def __init__(self):

        self.data = {}

    def count(self):

        return len(self.data)

    def upsert(
        self,
        ids,
        documents,
        embeddings,
        metadatas
    ):

        for index, chunk_id in enumerate(ids):

            self.data[chunk_id] = {
                "document": documents[index],
                "embedding": embeddings[index],
                "metadata": metadatas[index]
            }

    def get(self, where=None, include=None):

        selected = []

        for chunk_id, item in self.data.items():

            if where:

                matches = all(
                    item["metadata"].get(key) == value
                    for key, value in where.items()
                )

                if not matches:
                    continue

            selected.append((chunk_id, item))

        return {
            "ids": [item[0] for item in selected],
            "documents": [
                item[1]["document"] for item in selected
            ],
            "metadatas": [
                item[1]["metadata"] for item in selected
            ]
        }

    def delete(self, ids):

        for chunk_id in ids:
            self.data.pop(chunk_id, None)

    def query(
        self,
        query_embeddings,
        n_results,
        include
    ):

        selected = list(self.data.items())[:n_results]

        return {
            "ids": [
                item[0] for item in selected
            ],
            "documents": [
                item[1]["document"] for item in selected
            ],
            "metadatas": [
                item[1]["metadata"] for item in selected
            ],
            "distances": [
                0.1 for item in selected
            ]
        }


# ==========================================
# FAKE CHROMADB CLIENT
# ==========================================

class FakeClient:

    def __init__(self):

        self.collection = FakeCollection()

    def get_or_create_collection(
        self,
        name,
        metadata
    ):

        return self.collection


# ==========================================
# FIXTURE
# ==========================================

@pytest.fixture
def vector_store(tmp_path, monkeypatch):

    monkeypatch.chdir(tmp_path)

    fake_client = FakeClient()

    with patch(
        "src.vector_store.chromadb.PersistentClient",
        return_value=fake_client
    ):

        store = VectorStore(
            collection_name="test_collection",
            batch_size=2
        )

        yield store


# ==========================================
# 1. INITIALIZATION TESTS
# ==========================================

def test_vector_store_initialization(vector_store):

    assert vector_store.collection is not None

    assert vector_store.batch_size == 2

    assert vector_store.embedding_dimension is None


def test_database_directory_creation(vector_store):

    assert vector_store.db_path.exists()

    assert vector_store.db_path.is_dir()


# ==========================================
# 2. DOCUMENT ID TESTS
# ==========================================

def test_generate_document_id(vector_store):

    result = vector_store.generate_document_id(
        "sample.pdf"
    )

    assert isinstance(result, str)

    assert len(result) == 24


def test_document_id_consistency(vector_store):

    first = vector_store.generate_document_id(
        "sample.pdf"
    )

    second = vector_store.generate_document_id(
        "sample.pdf"
    )

    assert first == second


def test_document_id_case_insensitive(vector_store):

    first = vector_store.generate_document_id(
        "Sample.pdf"
    )

    second = vector_store.generate_document_id(
        "sample.pdf"
    )

    assert first == second


# ==========================================
# 3. CHUNK EXTRACTION TESTS
# ==========================================

def test_get_chunk_text_from_string(vector_store):

    result = vector_store.get_chunk_text(
        "  Artificial Intelligence  "
    )

    assert result == "Artificial Intelligence"


def test_get_chunk_text_from_dictionary(vector_store):

    result = vector_store.get_chunk_text({
        "text": "Machine Learning"
    })

    assert result == "Machine Learning"


def test_get_chunk_text_content_fallback(vector_store):

    result = vector_store.get_chunk_text({
        "content": "Deep Learning"
    })

    assert result == "Deep Learning"


def test_get_chunk_text_invalid_format(vector_store):

    with pytest.raises(ValueError):

        vector_store.get_chunk_text(123)


def test_get_page_number(vector_store):

    assert vector_store.get_page_number({
        "page_number": "5"
    }) == 5


def test_invalid_page_number_defaults_to_zero(vector_store):

    assert vector_store.get_page_number({
        "page_number": "invalid"
    }) == 0


# ==========================================
# 4. EMBEDDING VALIDATION TESTS
# ==========================================

def test_valid_embedding(vector_store):

    result = vector_store.validate_embedding(
        [0.5, 0.3, 0.8],
        0
    )

    assert result == [0.5, 0.3, 0.8]


@pytest.mark.parametrize(
    "embedding",
    [
        [],
        [0, 0, 0],
        [float("nan"), 0.5],
        [float("inf"), 0.5],
        ["invalid", 0.5]
    ]
)
def test_invalid_embeddings(vector_store, embedding):

    with pytest.raises(ValueError):

        vector_store.validate_embedding(
            embedding,
            0
        )


# ==========================================
# 5. ADD DOCUMENT TESTS
# ==========================================

def test_add_documents(vector_store):

    chunks = [
        {
            "text": "Introduction to AI",
            "page_number": 1
        },
        {
            "text": "Introduction to ML",
            "page_number": 2
        }
    ]

    embeddings = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6]
    ]

    vector_store.add_documents(
        chunks,
        embeddings,
        filename="sample.pdf",
        document_id="doc_001"
    )

    assert vector_store.get_collection_count() == 2

    assert vector_store.embedding_dimension == 3


def test_document_metadata_storage(vector_store):

    chunks = [
        {
            "text": "Test document",
            "page_number": 4
        }
    ]

    embeddings = [
        [0.1, 0.2, 0.3]
    ]

    vector_store.add_documents(
        chunks,
        embeddings,
        filename="sample.pdf",
        document_id="doc_002"
    )

    result = vector_store.get_all_chunks()

    assert result[0]["metadata"]["filename"] == "sample.pdf"

    assert result[0]["metadata"]["page_number"] == 4

    assert result[0]["metadata"]["document_id"] == "doc_002"


def test_chunk_embedding_count_mismatch(vector_store):

    with pytest.raises(ValueError):

        vector_store.add_documents(
            [{"text": "AI"}],
            [],
            document_id="doc_003"
        )


def test_empty_chunk_rejected(vector_store):

    with pytest.raises(ValueError):

        vector_store.add_documents(
            [{"text": "   "}],
            [[0.1, 0.2]],
            document_id="doc_004"
        )


def test_inconsistent_embedding_dimensions(vector_store):

    chunks = [
        {"text": "First"},
        {"text": "Second"}
    ]

    embeddings = [
        [0.1, 0.2],
        [0.1, 0.2, 0.3]
    ]

    with pytest.raises(ValueError):

        vector_store.add_documents(
            chunks,
            embeddings,
            document_id="doc_005"
        )


def test_empty_document_list(vector_store):

    vector_store.add_documents(
        [],
        [],
        document_id="doc_006"
    )

    assert vector_store.get_collection_count() == 0


# ==========================================
# 6. BATCH INSERTION TESTS
# ==========================================

def test_batch_insertion(vector_store):

    chunks = [
        {"text": "First"},
        {"text": "Second"},
        {"text": "Third"},
        {"text": "Fourth"},
        {"text": "Fifth"}
    ]

    embeddings = [
        [0.1, 0.2, 0.3],
        [0.2, 0.3, 0.4],
        [0.3, 0.4, 0.5],
        [0.4, 0.5, 0.6],
        [0.5, 0.6, 0.7]
    ]

    vector_store.add_documents(
        chunks,
        embeddings,
        document_id="batch_doc"
    )

    assert vector_store.get_collection_count() == 5


# ==========================================
# 7. DOCUMENT REPLACEMENT TESTS
# ==========================================

def test_document_replacement_removes_old_chunks(vector_store):

    first_chunks = [
        {"text": "First"},
        {"text": "Second"},
        {"text": "Third"}
    ]

    first_embeddings = [
        [0.1, 0.2],
        [0.2, 0.3],
        [0.3, 0.4]
    ]

    vector_store.add_documents(
        first_chunks,
        first_embeddings,
        document_id="replace_doc"
    )

    second_chunks = [
        {"text": "Updated document"}
    ]

    second_embeddings = [
        [0.5, 0.6]
    ]

    vector_store.add_documents(
        second_chunks,
        second_embeddings,
        document_id="replace_doc"
    )

    assert vector_store.get_collection_count() == 1

    result = vector_store.get_all_chunks()

    assert result[0]["content"] == "Updated document"


# ==========================================
# 8. SEARCH TESTS
# ==========================================

def test_search_empty_database(vector_store):

    result = vector_store.search(
        [0.1, 0.2, 0.3]
    )

    assert result == []


def test_search_returns_results(vector_store):

    vector_store.add_documents(
        [{"text": "Artificial Intelligence"}],
        [[0.1, 0.2, 0.3]],
        document_id="search_doc"
    )

    result = vector_store.search(
        [0.1, 0.2, 0.3]
    )

    assert len(result) == 1

    assert result[0]["content"] == "Artificial Intelligence"

    assert "distance" in result[0]

    assert "metadata" in result[0]


def test_search_limits_results(vector_store):

    chunks = [
        {"text": "First"},
        {"text": "Second"},
        {"text": "Third"}
    ]

    embeddings = [
        [0.1, 0.2],
        [0.2, 0.3],
        [0.3, 0.4]
    ]

    vector_store.add_documents(
        chunks,
        embeddings,
        document_id="search_limit"
    )

    result = vector_store.search(
        [0.1, 0.2],
        n_results=2
    )

    assert len(result) == 2


def test_search_caps_results_to_database_count(vector_store):

    vector_store.add_documents(
        [{"text": "Only one chunk"}],
        [[0.1, 0.2]],
        document_id="one_doc"
    )

    result = vector_store.search(
        [0.1, 0.2],
        n_results=10
    )

    assert len(result) == 1


def test_search_invalid_result_count(vector_store):

    with pytest.raises(ValueError):

        vector_store.search(
            [0.1, 0.2],
            n_results=0
        )


def test_search_invalid_embedding(vector_store):

    with pytest.raises(ValueError):

        vector_store.search([0, 0, 0])


def test_search_dimension_mismatch(vector_store):

    vector_store.add_documents(
        [{"text": "AI"}],
        [[0.1, 0.2, 0.3]],
        document_id="dimension_doc"
    )

    with pytest.raises(ValueError):

        vector_store.search([0.1, 0.2])


# ==========================================
# 9. RETRIEVAL TESTS
# ==========================================

def test_get_all_chunks(vector_store):

    vector_store.add_documents(
        [
            {"text": "First chunk"},
            {"text": "Second chunk"}
        ],
        [
            [0.1, 0.2],
            [0.2, 0.3]
        ],
        document_id="retrieve_doc"
    )

    result = vector_store.get_all_chunks()

    assert len(result) == 2


def test_list_documents(vector_store):

    vector_store.add_documents(
        [
            {"text": "First"},
            {"text": "Second"}
        ],
        [
            [0.1, 0.2],
            [0.2, 0.3]
        ],
        filename="report.pdf",
        document_id="list_doc"
    )

    result = vector_store.list_documents()

    assert len(result) == 1

    assert result[0]["filename"] == "report.pdf"

    assert result[0]["chunks"] == 2


# ==========================================
# 10. DELETE TESTS
# ==========================================

def test_delete_document(vector_store):

    vector_store.add_documents(
        [
            {"text": "First"},
            {"text": "Second"}
        ],
        [
            [0.1, 0.2],
            [0.2, 0.3]
        ],
        document_id="delete_doc"
    )

    result = vector_store.delete_document(
        "delete_doc"
    )

    assert result is True

    assert vector_store.get_collection_count() == 0


def test_delete_nonexistent_document(vector_store):

    result = vector_store.delete_document(
        "missing_doc"
    )

    assert result is False


def test_collection_count(vector_store):

    assert vector_store.get_collection_count() == 0

    vector_store.add_documents(
        [{"text": "Test"}],
        [[0.1, 0.2]],
        document_id="count_doc"
    )

    assert vector_store.get_collection_count() == 1
