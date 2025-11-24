import time
import pytest
from app.rag.service import RAGService


@pytest.mark.integration
def test_retrieve_latency():
    """
    Verifies retrieve() completes in reasonable time for single query.
    Note: Comprehensive latency testing (≤300ms median) is in test_rag_evaluation.py
    """
    service = RAGService()

    start = time.perf_counter()
    result = service.retrieve("test query")
    duration = (time.perf_counter() - start) * 1000

    # Smoke test: single query should complete within 1 second
    assert duration < 1000, f"Retrieve took {duration:.0f}ms (expected <1000ms)"
    assert isinstance(result, list), "Retrieve should return a list of results"
