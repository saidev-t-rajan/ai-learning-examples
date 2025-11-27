from functools import lru_cache
from typing import cast
from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


def embed_query(text: str) -> list[float]:
    return cast(list[float], _get_model().encode(text).tolist())


def embed_documents(texts: list[str]) -> list[list[float]]:
    return cast(list[list[float]], _get_model().encode(texts).tolist())


SPECIAL_TOKEN_COUNT = 2


def get_token_count(text: str) -> int:
    model = _get_model()
    if not text:
        return SPECIAL_TOKEN_COUNT

    return len(model.tokenizer.encode(text, truncation=False, add_special_tokens=True))
