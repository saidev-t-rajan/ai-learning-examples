import json
import statistics
import time
import os
import pytest
from app.rag.service import RAGService

# Path to the evaluation set
EVAL_SET_PATH = os.path.join(os.path.dirname(__file__), "data", "evaluation_set.json")


@pytest.mark.integration
def test_rag_accuracy_and_latency():
    """
    Evaluates the RAG system's retrieval accuracy and latency against a predefined set of questions.
    Fails if Top-5 accuracy is below 70% or median latency is above 300ms.
    """
    if not os.path.exists(EVAL_SET_PATH):
        pytest.fail(f"Evaluation set not found at {EVAL_SET_PATH}")

    with open(EVAL_SET_PATH, "r") as f:
        questions = json.load(f)

    # Initialize Service
    # Note: This assumes the vector store is already populated with the relevant documents.
    # If running in a fresh CI environment, you might need to run an ingestion step fixture here.
    rag_service = RAGService()

    latencies = []
    success_count = 0
    failures = []

    for i, item in enumerate(questions, 1):
        q = item["question"]
        expected_keywords = item["expected_keywords"]

        # Measure Latency
        start = time.perf_counter()
        # We retrieve top 10 results but check only top 5 for accuracy
        # Calling vector_store directly to get the list of results
        results = rag_service.vector_store.similarity_search(
            query=q, k=10, embedding_service=rag_service.embeddings
        )
        end = time.perf_counter()

        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)

        # Check Accuracy (Top-5)
        # We take the first 5 results from the 10 retrieved
        top_5_results = results[:5]

        # Flatten text from results to check for keywords
        # Results can be strings or tuples (text, metadata)
        combined_text_top_5 = ""
        for res in top_5_results:
            if isinstance(res, tuple):
                combined_text_top_5 += res[0] + "\n"
            else:
                combined_text_top_5 += res + "\n"

        lower_result = combined_text_top_5.lower()

        # We consider it a "Hit" if ANY of the expected keywords are present in the top 5 chunks
        hits = [k for k in expected_keywords if k.lower() in lower_result]
        is_success = len(hits) > 0

        if is_success:
            success_count += 1
        else:
            failures.append(
                {
                    "question": q,
                    "expected": expected_keywords,
                    "got_snippet": combined_text_top_5[:200].replace("\n", " ") + "...",
                }
            )

    # Metrics
    accuracy = (success_count / len(questions)) * 100 if questions else 0
    median_latency = statistics.median(latencies) if latencies else 0
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0

    # Print Report for valid pytest output (use -s to see)
    print("\n" + "=" * 30)
    print("EVALUATION REPORT (Top-5)")
    print("=" * 30)
    print(f"Total Questions: {len(questions)}")
    print(f"Top-5 Accuracy:  {accuracy:.1f}%")
    print(f"Median Latency:  {median_latency:.1f}ms")
    print(f"P95 Latency:     {p95_latency:.1f}ms")
    if failures:
        print("-" * 30)
        print("FAILURES:")
        for fail in failures:
            print(f"Q: {fail['question']}")
            print(f"   Expected: {fail['expected']}")
            print(f"   Got (Top 5): {fail['got_snippet']}")
    print("=" * 30)

    # Assertions
    assert accuracy >= 70, f"Accuracy {accuracy:.1f}% is below threshold of 70%"
    assert median_latency <= 300, (
        f"Median Latency {median_latency:.1f}ms is above threshold of 300ms"
    )
