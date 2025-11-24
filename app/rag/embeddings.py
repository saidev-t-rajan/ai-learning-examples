from functools import lru_cache
from typing import cast
from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Lazy-load the embedding model (cached singleton)."""
    print("Notice: Using local 'all-MiniLM-L6-v2' embeddings.\n")
    return SentenceTransformer("all-MiniLM-L6-v2")


def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    return cast(list[float], _get_model().encode(text).tolist())


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed a list of documents."""
    return cast(list[list[float]], _get_model().encode(texts).tolist())
