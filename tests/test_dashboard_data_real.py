import pytest
import os
from app.db.chat_repository import ChatRepository
from app.db.vector import ChromaVectorStore
from app.rag.service import RAGService
from app.core.models import ChatMetrics


@pytest.fixture
def clean_db(tmp_path):
    """Provide a fresh DB path."""
    db_path = tmp_path / "test_chat.db"
    return str(db_path)


@pytest.fixture
def clean_chroma(tmp_path):
    """Provide a fresh Chroma persistence directory."""
    chroma_dir = tmp_path / "chroma_test"
    return str(chroma_dir)


def test_repo_schema_v2_metrics(clean_db):
    """Test that we can store and retrieve V2 metrics."""
    repo = ChatRepository(db_path=clean_db)

    # Add an assistant message with metrics
    input_metrics = ChatMetrics(
        input_tokens=10,
        output_tokens=20,
        cost=0.005,
        total_latency=1.5,
        ttft=0.5,
        avg_retrieval_distance=0.8,
        rag_success=True,
        response_status="success",
    )
    repo.add_message(
        role="assistant",
        content="Test content",
        metrics=input_metrics,
    )

    # Retrieve stats
    metrics = repo.get_assistant_metrics(limit=1)
    assert len(metrics) == 1
    entry = metrics[0]
    m = entry.metrics

    assert m.input_tokens == 10
    assert m.output_tokens == 20
    assert abs(m.cost - 0.005) < 0.0001
    assert abs(m.total_latency - 1.5) < 0.0001
    assert abs(m.ttft - 0.5) < 0.0001
    assert m.avg_retrieval_distance is not None
    assert abs(m.avg_retrieval_distance - 0.8) < 0.0001
    assert m.rag_success is True
    assert m.response_status == "success"


def test_repo_success_breakdown(clean_db):
    """Test the pie chart data aggregation."""
    repo = ChatRepository(db_path=clean_db)

    # 1. Success with RAG
    repo.add_message(
        "assistant",
        "OK",
        metrics=ChatMetrics(rag_success=True, response_status="success"),
    )
    # 2. Success without RAG (partial/no context)
    repo.add_message(
        "assistant",
        "OK",
        metrics=ChatMetrics(rag_success=False, response_status="success"),
    )
    # 3. Error
    repo.add_message(
        "assistant",
        "Err",
        metrics=ChatMetrics(rag_success=False, response_status="error:timeout"),
    )

    breakdown = repo.get_success_breakdown()
    assert breakdown["full_success"] == 1
    assert breakdown["partial"] == 1
    assert breakdown["error"] == 1


def test_rag_returns_distances(clean_chroma):
    """Test that RAG service returns distance scores."""
    # Setup Vector Store
    store = ChromaVectorStore(persist_directory=clean_chroma)
    rag = RAGService(vector_store=store)

    # Ingest a dummy document
    test_file = os.path.join(os.path.dirname(clean_chroma), "test.txt")
    with open(test_file, "w") as f:
        f.write("This is a test document about bananas.")

    rag.ingest(test_file)

    # Search
    results = rag.retrieve("bananas")
    assert len(results) > 0
    text, meta, dist = results[0]

    assert "bananas" in text
    assert isinstance(dist, float)
    # Distance should be small for exact match content (though embeddings vary)
    # Chroma default is usually L2 or Cosine. If Cosine distance, range 0-2.
    # If Inner Product, varies. default is usually l2 squared?
    # Whatever it is, it should be a float.
    assert dist >= 0.0
