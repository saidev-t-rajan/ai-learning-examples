import json
import time
import os
import sys
import statistics

# Ensure app is in path (add parent directory of 'tests' which is 'ai-learning-examples')
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))

from app.rag.service import RAGService

# Path relative to this script
EVAL_SET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "evaluation_set.json"
)


def evaluate():
    if not os.path.exists(EVAL_SET_PATH):
        print(f"Error: {EVAL_SET_PATH} not found.")
        return

    with open(EVAL_SET_PATH, "r") as f:
        questions = json.load(f)

    print(f"Starting evaluation with {len(questions)} questions...")

    # Initialize Service
    # Note: This assumes the vector store is already populated (Task 6.5)
    rag_service = RAGService()

    latencies = []
    success_count = 0

    for i, item in enumerate(questions, 1):
        q = item["question"]
        expected_keywords = item["expected_keywords"]

        print(f"\nQ{i}: {q}")

        # Measure Latency
        start = time.perf_counter()
        result = rag_service.retrieve(q)
        end = time.perf_counter()

        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)

        # Check Accuracy (Simple keyword match in retrieved context)
        # result is a string of joined chunks
        lower_result = result.lower()

        # We consider it a "Hit" if ANY of the expected keywords are present
        # Or should it be ALL? "Any" is usually safer for RAG retrieval checks unless very specific.
        # Let's go with "Any" for now, or "All" if the keyword is unique.
        # Let's stick to "Any" of the keywords list found.

        hits = [k for k in expected_keywords if k.lower() in lower_result]
        is_success = len(hits) > 0

        if is_success:
            success_count += 1
            print(f"  ✅ Success ({latency_ms:.1f}ms)")
        else:
            print(f"  ❌ Failed ({latency_ms:.1f}ms)")
            print(f"     Expected: {expected_keywords}")
            print(
                f"     Got: {result[:200].replace(chr(10), ' ')}..."
            )  # Print snippet for debug

    # Report
    accuracy = (success_count / len(questions)) * 100
    median_latency = statistics.median(latencies)
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]

    print("\n" + "=" * 30)
    print("EVALUATION REPORT")
    print("=" * 30)
    print(f"Total Questions: {len(questions)}")
    print(f"Accuracy:        {accuracy:.1f}%")
    print(f"Median Latency:  {median_latency:.1f}ms")
    print(f"P95 Latency:     {p95_latency:.1f}ms")
    print("=" * 30)

    # Assertions for CI/CD
    if accuracy < 70:
        print("FAILED: Accuracy below 70%")
        exit(1)

    if median_latency > 300:
        print("FAILED: Median Latency > 300ms")
        exit(1)

    print("PASSED")


if __name__ == "__main__":
    evaluate()
