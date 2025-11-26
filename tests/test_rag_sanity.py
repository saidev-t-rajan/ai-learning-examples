import pytest
from app.rag.service import RAGService


@pytest.mark.integration
def test_manual_rag_sanity_check():
    """
    A sanity check to ensure the RAG service can retrieve basic context.
    Corresponds to the old debug_one.py script.
    """
    service = RAGService()
    query = "What is the name of the ship in Moby Dick?"

    # We execute the retrieval
    results = service.retrieve(query)

    # We log the result for manual inspection if needed (use -s to see)
    print(f"\nQuery: {query}")
    print("--- Result ---")
    print(results)
    print("--------------")

    # Combine all text from results for the check
    combined_text = " ".join([text for text, _, _ in results]).lower()

    is_relevant = (
        "pequod" in combined_text
        or "ahab" in combined_text
        or "moby dick" in combined_text
    )

    assert is_relevant, (
        "Expected 'Pequod', 'Ahab', or 'Moby Dick' in retrieval result. Have you ingested moby_dick.txt?"
    )
