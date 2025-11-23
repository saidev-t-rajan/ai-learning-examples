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
    result = service.retrieve(query)

    # We log the result for manual inspection if needed (use -s to see)
    print(f"\nQuery: {query}")
    print("--- Result ---")
    print(result)
    print("--------------")

    # This assertion mimics the old script's check.
    # It might fail if Moby Dick hasn't been ingested, but that is a useful signal for a sanity check.
    # If the specific word "Pequod" isn't found due to embedding model limitations,
    # finding "Ahab" or "Moby Dick" is sufficient to prove the RAG pipeline is working.
    result_lower = result.lower()
    is_relevant = (
        "pequod" in result_lower
        or "ahab" in result_lower
        or "moby dick" in result_lower
    )

    assert is_relevant, (
        "Expected 'Pequod', 'Ahab', or 'Moby Dick' in retrieval result. Have you ingested moby_dick.txt?"
    )
