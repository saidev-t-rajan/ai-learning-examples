from unittest.mock import MagicMock
from app.core.chat_service import ChatService
from app.db.memory import ChatRepository
from app.rag.protocol import RAGService


class MockRAGService(RAGService):
    def ingest(self, path: str) -> int:
        return 0

    def retrieve(self, query: str) -> str:
        # Simulate a formatted citation response
        return "[1] (Source: doc.pdf)\nThe sky is blue."


def test_system_prompt_contains_citation_instructions():
    # Setup
    repo = ChatRepository(db_path=":memory:")
    rag_service = MockRAGService()
    chat_service = ChatService(repo=repo, rag_service=rag_service)

    # We intercept the OpenAI client call to check the messages sent
    chat_service.client = MagicMock()

    # Act
    # We trigger the generator but don't iterate fully since we just want to check the call
    gen = chat_service.get_response("Why is the sky blue?")
    next(gen, None)  # Trigger execution

    # Assert
    call_args = chat_service.client.chat.completions.create.call_args
    assert call_args is not None

    kwargs = call_args.kwargs
    messages = kwargs["messages"]
    system_message = next(m for m in messages if m["role"] == "system")

    # The prompt must instruct the model about citations
    assert "Cite sources using the format [1], [2]" in system_message["content"]
    # And it should include our retrieved context
    assert "[1] (Source: doc.pdf)" in system_message["content"]
