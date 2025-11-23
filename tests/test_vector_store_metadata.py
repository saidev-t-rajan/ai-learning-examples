import shutil
import os
import pytest
from app.db.vector import ChromaVectorStore
from app.rag.embeddings import LocalEmbeddings


@pytest.fixture
def vector_store():
    # Setup: Use a temp dir for chroma
    temp_db = "data/test_chroma_metadata"
    if os.path.exists(temp_db):
        shutil.rmtree(temp_db)

    store = ChromaVectorStore(persist_directory=temp_db)
    yield store

    # Teardown
    if os.path.exists(temp_db):
        shutil.rmtree(temp_db)


def test_similarity_search_returns_metadata(vector_store):
    embeddings = LocalEmbeddings()

    # Act: Add documents with metadata
    vector_store.add_documents(
        texts=["Apple is a fruit."],
        metadatas=[{"source": "fruit_wiki.pdf"}],
        embedding_service=embeddings,
    )

    # Act: Search
    results = vector_store.similarity_search(
        query="Apple", k=1, embedding_service=embeddings
    )

    # Assert
    assert len(results) == 1
    first_result = results[0]

    # Should be a tuple (text, metadata)
    assert isinstance(first_result, tuple)
    assert first_result[0] == "Apple is a fruit."
    assert first_result[1] == {"source": "fruit_wiki.pdf"}
