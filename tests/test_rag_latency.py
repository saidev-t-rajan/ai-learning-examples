import logging
import time
from unittest.mock import MagicMock
from app.rag.service import RAGService
from app.db.vector import ChromaVectorStore


def test_retrieve_logs_warning_if_slow(caplog):
    # Setup
    mock_store = MagicMock(spec=ChromaVectorStore)

    # Mock similarity_search to sleep > 0.3s
    def slow_search(*args, **kwargs):
        time.sleep(0.31)
        return []

    mock_store.similarity_search.side_effect = slow_search

    service = RAGService(vector_store=mock_store)

    # Act
    with caplog.at_level(logging.WARNING):
        service.retrieve("slow query")

    # Assert
    # We expect a warning about execution time
    assert any(
        "took" in record.message and ">0.30s" in record.message
        for record in caplog.records
    )
