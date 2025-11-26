import pytest
from app.rag.service import RAGService
from app.db.vector import ChromaVectorStore


@pytest.mark.integration
def test_retrieve_returns_correct_metadata(tmp_path):
    # Setup - Use real components
    # Isolate DB
    store = ChromaVectorStore(
        collection_name="test_formatting", persist_directory=str(tmp_path / "chroma_db")
    )

    # Add documents manually to the store
    texts = ["The sky is blue.", "Roses are red."]
    metadatas = [{"source": "nature.pdf"}, {"source": "poetry.txt"}]
    store.add_documents(texts, metadatas)

    service = RAGService(vector_store=store)

    # Act
    # We query for something that matches the documents
    # "colors" might match "blue" and "red"
    results = service.retrieve("colors")

    # Assert
    # Verify we get a list of (text, metadata) tuples
    assert isinstance(results, list)

    # We expect to find our documents with their metadata
    found_nature = False
    found_poetry = False

    for text, meta, _ in results:
        if "The sky is blue." in text and meta.get("source") == "nature.pdf":
            found_nature = True
        if "Roses are red." in text and meta.get("source") == "poetry.txt":
            found_poetry = True

    assert found_nature, "Did not find nature document with correct metadata"
    assert found_poetry, "Did not find poetry document with correct metadata"
