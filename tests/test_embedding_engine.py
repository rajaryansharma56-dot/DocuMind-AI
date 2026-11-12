
import pytest
import numpy as np

from unittest.mock import patch

from src import embedding_engine


# ==========================================
# FIXTURE: MOCK EMBEDDING MODEL
# ==========================================

@pytest.fixture
def mock_model():

    with patch.object(embedding_engine, "model") as mocked:

        mocked.encode.side_effect = lambda texts, **kwargs: (
            np.array(
                [[1.0, 0.0, 0.0]]
            ) if isinstance(texts, str)
            else np.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0]
            ])
        )

        yield mocked


# ==========================================
# 1. SINGLE EMBEDDING TESTS
# ==========================================

def test_generate_embedding(mock_model):

    result = embedding_engine.generate_embedding(
        "DocuMind AI"
    )

    assert isinstance(result, np.ndarray)

    assert result.shape == (1, 3)

    mock_model.encode.assert_called_once()


def test_single_embedding_normalization(mock_model):

    embedding_engine.generate_embedding("Testing")

    kwargs = mock_model.encode.call_args.kwargs

    assert kwargs["normalize_embeddings"] is True
    assert kwargs["convert_to_numpy"] is True


def test_single_embedding_invalid_type():

    with pytest.raises(TypeError):

        embedding_engine.generate_embedding(123)


def test_single_embedding_empty_text():

    with pytest.raises(ValueError):

        embedding_engine.generate_embedding("   ")


# ==========================================
# 2. MULTIPLE EMBEDDING TESTS
# ==========================================

def test_generate_multiple_embeddings(mock_model):

    texts = [
        "Artificial Intelligence",
        "Machine Learning"
    ]

    result = embedding_engine.generate_embeddings(texts)

    assert isinstance(result, np.ndarray)

    assert result.shape == (2, 3)


def test_multiple_embedding_empty_list():

    with pytest.raises(ValueError):

        embedding_engine.generate_embeddings([])


def test_multiple_embedding_invalid_items():

    invalid_inputs = [
        ["AI", ""],
        ["AI", "   "],
        ["AI", 123],
        [None]
    ]

    for texts in invalid_inputs:

        with pytest.raises(ValueError):

            embedding_engine.generate_embeddings(texts)


def test_multiple_embedding_normalization(mock_model):

    embedding_engine.generate_embeddings(["AI", "ML"])

    kwargs = mock_model.encode.call_args.kwargs

    assert kwargs["normalize_embeddings"] is True
    assert kwargs["convert_to_numpy"] is True


# ==========================================
# 3. EMBEDDING INFORMATION TESTS
# ==========================================

def test_embedding_information():

    embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ])

    result = embedding_engine.get_embedding_info(embeddings)

    assert result["number_of_embeddings"] == 2
    assert result["embedding_dimensions"] == 3
    assert result["data_type"] == "float64"


def test_embedding_information_empty():

    with pytest.raises(ValueError):

        embedding_engine.get_embedding_info([])


def test_embedding_information_single_vector():

    embeddings = np.array([1.0, 2.0, 3.0])

    result = embedding_engine.get_embedding_info(embeddings)

    assert result["number_of_embeddings"] == 3
    assert result["embedding_dimensions"] == 3


# ==========================================
# 4. NUMERICAL VALIDATION
# ==========================================

def test_embedding_values_are_finite(mock_model):

    result = embedding_engine.generate_embedding("Testing")

    assert np.all(np.isfinite(result))


def test_embedding_dimensions(mock_model):

    result = embedding_engine.generate_embeddings([
        "Artificial Intelligence",
        "Deep Learning"
    ])

    assert result.shape[1] == 3
