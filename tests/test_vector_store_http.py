import os

import pytest

from app.db.vector import ChromaVectorStore


@pytest.mark.integration
def test_http_client_mode_add_and_search() -> None:
    """
    Test ChromaVectorStore in HTTP client mode.

    Host and port are taken from the environment to support both
    local runs (default: localhost:8000) and docker-compose where
    CHROMA_HOST=chromadb, CHROMA_PORT=8000.

    For a standalone local server, you can run:
      docker run -p 8000:8000 ghcr.io/chroma-core/chroma:latest
    """
    host = os.getenv("CHROMA_HOST", "localhost")
    port_str = os.getenv("CHROMA_PORT", "8000")
    port = int(port_str)

    store = ChromaVectorStore(
        host=host,
        port=port,
        collection_name="test_http_integration",
    )

    texts = ["The quick brown fox jumps over the lazy dog."]
    metadatas = [{"source": "test_http.txt"}]

    store.add_documents(texts=texts, metadatas=metadatas)

    results = store.similarity_search(query="fox jumping", k=1)

    assert len(results) == 1
    text, metadata, distance = results[0]
    assert "fox" in text.lower()
    assert metadata["source"] == "test_http.txt"
    assert isinstance(distance, float)

    store.client.delete_collection(name="test_http_integration")


def test_embedded_mode_requires_persist_directory() -> None:
    with pytest.raises(ValueError, match="persist_directory is required"):
        ChromaVectorStore(persist_directory=None, host=None)
