from functools import lru_cache
from typing import cast
from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Lazy-load the embedding model (cached singleton)."""
    return SentenceTransformer("all-MiniLM-L6-v2")


def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    return cast(list[float], _get_model().encode(text).tolist())


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed a list of documents."""
    return cast(list[list[float]], _get_model().encode(texts).tolist())


# [CLS] and [SEP] tokens for BERT-based models
SPECIAL_TOKEN_COUNT = 2


def get_token_count(text: str) -> int:
    """Count tokens using the embedding model's native tokenizer."""
    model = _get_model()
    # Handle edge case: empty string
    if not text:
        return SPECIAL_TOKEN_COUNT

    return len(model.tokenizer.encode(text, truncation=False, add_special_tokens=True))
