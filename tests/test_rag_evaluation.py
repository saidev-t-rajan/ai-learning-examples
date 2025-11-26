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
    Evaluates RAG retrieval accuracy and latency, split by question type.

    Question types:
    - Factual: Specific facts/entities requiring keyword matching
    - Comprehension: Context/theme questions benefiting from better chunking

    Requirements: Overall accuracy ≥65%, median latency ≤300ms (Task 3.2.c.ii, 3.2.d)
    """
    with open(EVAL_SET_PATH) as f:
        questions = json.load(f)

    rag_service = RAGService()

    # Track metrics by question type
    latencies: list[float] = []
    factual_success = 0
    factual_total = 0
    factual_failures: list[tuple[str, list[str]]] = []
    comprehension_success = 0
    comprehension_total = 0
    comprehension_failures: list[tuple[str, list[str]]] = []

    for item in questions:
        question = item["question"]
        expected_keywords = item["expected_keywords"]
        question_type = item.get("question_type", "factual")  # Default to factual

        # Measure retrieval latency (vector search only, excludes LLM)
        start = time.perf_counter()
        results = rag_service.vector_store.similarity_search(query=question, k=5)
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)

        # Extract text from results (handles tuple or string format)
        texts = [r[0] if isinstance(r, tuple) else r for r in results]
        combined_text = " ".join(texts).lower()

        # Check if any expected keyword appears in top-5 results
        success = any(keyword.lower() in combined_text for keyword in expected_keywords)

        # Track by question type
        if question_type == "factual":
            factual_total += 1
            if success:
                factual_success += 1
            else:
                factual_failures.append((question, expected_keywords))
        else:  # comprehension
            comprehension_total += 1
            if success:
                comprehension_success += 1
            else:
                comprehension_failures.append((question, expected_keywords))

    # Calculate metrics
    total_success = factual_success + comprehension_success
    total_questions = len(questions)
    overall_accuracy = (total_success / total_questions) * 100

    factual_accuracy = (
        (factual_success / factual_total) * 100 if factual_total > 0 else 0
    )
    comprehension_accuracy = (
        (comprehension_success / comprehension_total) * 100
        if comprehension_total > 0
        else 0
    )

    median_latency = statistics.median(latencies)
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]

    # Report results
    print(f"\n{'=' * 70}")
    print(f"RAG EVALUATION: {total_questions} questions")
    print(f"{'=' * 70}")
    print(
        f"Overall Accuracy:        {overall_accuracy:.1f}% ({total_success}/{total_questions})"
    )
    print(
        f"  Factual Questions:     {factual_accuracy:.1f}% ({factual_success}/{factual_total})"
    )
    print(
        f"  Comprehension Qs:      {comprehension_accuracy:.1f}% ({comprehension_success}/{comprehension_total})"
    )
    print("")
    print(f"Latency:  Median={median_latency:.0f}ms | P95={p95_latency:.0f}ms")
    print(f"{'=' * 70}")

    # Show failures by type
    if factual_failures:
        print(f"\nFactual Failures ({len(factual_failures)}):")
        for q, keywords in factual_failures[:3]:
            print(f"  - {q[:55]}... (expected: {keywords})")

    if comprehension_failures:
        print(f"\nComprehension Failures ({len(comprehension_failures)}):")
        for q, keywords in comprehension_failures[:3]:
            print(f"  - {q[:55]}... (expected: {keywords})")

    print("=" * 70)

    # Assert requirements
    assert overall_accuracy >= 65, (
        f"Overall accuracy {overall_accuracy:.1f}% below 65% threshold"
    )
    assert median_latency <= 300, (
        f"Median latency {median_latency:.0f}ms above 300ms threshold"
    )
