import pytest
from app.rag.service import RAGService
from app.db.vector import ChromaVectorStore


@pytest.mark.integration
def test_rag_ingest_and_retrieve_integration(tmp_path):
    # Setup a dummy text file
    # We use a .txt file for simplicity in testing.
    # The RAGService should be able to handle it (or we will implement a TextLoader).
    d = tmp_path / "data"
    d.mkdir()
    p = d / "test_doc.txt"
    content = "The secret code is BLUE-HORIZON-99."
    p.write_text(content)

    # Create an isolated vector store using tmp_path
    isolated_store = ChromaVectorStore(
        collection_name="test_rag_integration",
        persist_directory=str(tmp_path / "chroma_db"),
    )

    service = RAGService(vector_store=isolated_store)

    # Ingest
    # Expecting it to return number of chunks or documents
    count = service.ingest(str(p))
    assert count > 0

    # Retrieve
    # We expect to find the content we just ingested
    results = service.retrieve("What is the secret code?")

    # Check if any result contains the secret code
    found = any("BLUE-HORIZON-99" in text for text, _ in results)
    assert found, "Secret code not found in retrieval results"
