
from sentence_transformers import SentenceTransformer
import numpy as np


# Load the embedding model
MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)


def generate_embedding(text):
    """
    Generate an embedding for a single text.
    """

    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    if not text.strip():
        raise ValueError("Text cannot be empty.")

    embedding = model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embedding


def generate_embeddings(texts):
    """
    Generate embeddings for multiple text chunks.
    """

    if not texts:
        raise ValueError("The text list cannot be empty.")

    if not all(isinstance(text, str) and text.strip() for text in texts):
        raise ValueError("Every item must be a non-empty string.")

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return embeddings


def get_embedding_info(embeddings):
    """
    Return basic information about generated embeddings.
    """

    embeddings = np.asarray(embeddings)

    if embeddings.size == 0:
        raise ValueError("Embeddings cannot be empty.")

    return {
        "number_of_embeddings": int(embeddings.shape[0]),
        "embedding_dimensions": int(embeddings.shape[-1]),
        "data_type": str(embeddings.dtype)
    }
