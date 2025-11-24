import json
import statistics
import time
import os
import pytest
from app.rag.service import RAGService

EVAL_SET_PATH = os.path.join(os.path.dirname(__file__), "data", "evaluation_set.json")


@pytest.mark.integration
def test_rag_accuracy_and_latency():
    """
    Evaluates RAG retrieval accuracy and latency on 20 test questions.
    Requirements: Top-5 accuracy ≥65%, median latency ≤300ms (Task 3.2.c.ii, 3.2.d)
    """
    with open(EVAL_SET_PATH) as f:
        questions = json.load(f)

    rag_service = RAGService()
    latencies = []
    success_count = 0
    failures = []

    for item in questions:
        question = item["question"]
        expected_keywords = item["expected_keywords"]

        # Measure retrieval latency (vector search only, excludes LLM)
        start = time.perf_counter()
        results = rag_service.vector_store.similarity_search(query=question, k=5)
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)

        # Extract text from results (handles tuple or string format)
        texts = [r[0] if isinstance(r, tuple) else r for r in results]
        combined_text = " ".join(texts).lower()

        # Check if any expected keyword appears in top-5 results
        if any(keyword.lower() in combined_text for keyword in expected_keywords):
            success_count += 1
        else:
            failures.append((question, expected_keywords))

    # Calculate metrics
    accuracy = (success_count / len(questions)) * 100
    median_latency = statistics.median(latencies)
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]

    # Report results
    print(f"\n{'=' * 50}")
    print(f"RAG EVALUATION: {len(questions)} questions")
    print(
        f"Top-5 Accuracy: {accuracy:.1f}% | Median: {median_latency:.0f}ms | P95: {p95_latency:.0f}ms"
    )
    if failures:
        print(f"Failed {len(failures)} questions:")
        for q, keywords in failures[:3]:  # Show first 3 failures
            print(f"  - {q[:60]}... (expected: {keywords})")
    print("=" * 50)

    # Assert requirements
    assert accuracy >= 65, f"Accuracy {accuracy:.1f}% below 65% threshold"
    assert median_latency <= 300, (
        f"Median latency {median_latency:.0f}ms above 300ms threshold"
    )
