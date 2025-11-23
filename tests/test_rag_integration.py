import pytest
from app.rag.service import RAGService


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

    service = RAGService()

    # Ingest
    # Expecting it to return number of chunks or documents
    count = service.ingest(str(p))
    assert count > 0

    # Retrieve
    # We expect to find the content we just ingested
    results = service.retrieve("What is the secret code?")
    assert "BLUE-HORIZON-99" in results
