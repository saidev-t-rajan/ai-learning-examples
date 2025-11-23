import logging
import time
from app.rag.service import RAGService
from app.db.vector import ChromaVectorStore


class SlowVectorStore(ChromaVectorStore):
    def similarity_search(self, query, k=4, embedding_service=None):
        time.sleep(0.31)  # Force latency
        return []


def test_retrieve_logs_warning_if_slow(caplog, tmp_path):
    # Setup
    # We pass a dummy persist directory
    store = SlowVectorStore(
        collection_name="test_slow", persist_directory=str(tmp_path / "chroma_db")
    )

    # We don't even need real embeddings since we override similarity_search
    # but RAGService expects it to work.

    service = RAGService(vector_store=store)

    # Act
    with caplog.at_level(logging.WARNING):
        service.retrieve("slow query")

    # Assert
    # We expect a warning about execution time
    assert any(
        "took" in record.message and ">0.30s" in record.message
        for record in caplog.records
    )
