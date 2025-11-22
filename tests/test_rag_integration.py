import pytest
from app.rag.mock_service import MockRAGService
from app.core.chat_service import ChatService
from app.db.memory import ChatRepository


@pytest.fixture
def repo():
    return ChatRepository(db_path=":memory:")


def test_ingest_flow():
    """
    Test that we can ingest a file via the RAG service.
    """
    rag_service = MockRAGService()
    count = rag_service.ingest("some/path/to/doc.pdf")
    assert count == 5


@pytest.mark.integration
def test_retrieval_in_chat(repo):
    """
    Test that the ChatService utilizes the RAG service to augment the prompt.
    We allow real API calls here.
    """
    rag_service = MockRAGService()
    chat_service = ChatService(repo=repo, rag_service=rag_service)

    # The MockRAGService always returns "Context: This is the retrieved context."
    # We ask a question that specifically relies on this context to verify it was injected.
    user_message = "What does the context say?"

    response_gen = chat_service.get_response(user_message)

    full_response = ""
    for chunk in response_gen:
        if isinstance(chunk, str):
            full_response += chunk

    # We expect the LLM to see the context and answer based on it.
    # Since the mock context is "This is the retrieved context", the LLM should mention "retrieved context".
    assert (
        "retrieved context" in full_response.lower()
        or "context" in full_response.lower()
    )

    # Also verify the RAG service was actually queried (spy check)
    assert rag_service.last_query == user_message
