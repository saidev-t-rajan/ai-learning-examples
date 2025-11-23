import pytest
from app.rag.service import RAGService
from app.db.vector import ChromaVectorStore
from app.rag.embeddings import LocalEmbeddings


@pytest.mark.integration
def test_retrieve_formats_citations(tmp_path):
    # Setup - Use real components
    embeddings = LocalEmbeddings()

    # Isolate DB
    store = ChromaVectorStore(
        collection_name="test_formatting", persist_directory=str(tmp_path / "chroma_db")
    )

    # Add documents manually to the store
    texts = ["The sky is blue.", "Roses are red."]
    metadatas = [{"source": "nature.pdf"}, {"source": "poetry.txt"}]
    store.add_documents(texts, metadatas, embedding_service=embeddings)

    service = RAGService(vector_store=store)

    # Act
    # We query for something that matches the documents
    # "colors" might match "blue" and "red"
    result = service.retrieve("colors")

    # Assert
    # The order might vary, so we check for existence of formatted strings
    expected_part_1 = "(Source: nature.pdf)\nThe sky is blue."
    expected_part_2 = "(Source: poetry.txt)\nRoses are red."

    assert expected_part_1 in result
    assert expected_part_2 in result
