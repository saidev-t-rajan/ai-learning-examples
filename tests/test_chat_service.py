import pytest

# TDD: Import will fail initially
try:
    from app.core.chat_service import ChatService
except ImportError:
    ChatService = None


def test_chat_service_structure(settings, repo):
    """
    Test that the ChatService class exists and has a get_response method.
    """
    if ChatService is None:
        pytest.fail("app.core.chat_service module not found")

    service = ChatService(repo=repo, settings=settings)
    assert hasattr(service, "get_response")


@pytest.mark.integration
def test_conversation_memory_real(settings, repo):
    """
    Test that the conversation state is actually saved and retrieved using REAL components.
    This invokes the REAL OpenAI API, but we check the LOCAL side effects (DB persistence).
    """
    service = ChatService(repo=repo, settings=settings)

    # 1. Send a message
    # We use a very short prompt to save tokens/time
    service.get_response("Hi")

    # 2. Check DB directly to verify persistence
    history = repo.get_recent_messages(limit=10)
    # Expect: User("Hi") + Assistant(Response)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Hi"
    assert history[1]["role"] == "assistant"
    assert len(history[1]["content"]) > 0

    # 3. Send another message to verify context
    # We can't easily prove the model "remembered" without a complex prompt,
    # but we can verify the DB grew.
    service.get_response("Bye")

    history_new = repo.get_recent_messages(limit=10)
    # Expect: User("Hi") + Assistant(Response1) + User("Bye") + Assistant(Response2)
    assert len(history_new) == 4
    assert history_new[2]["role"] == "user"
    assert history_new[2]["content"] == "Bye"


@pytest.mark.integration
def test_get_response_real_api(settings, repo):
    """
    Test that get_response calls the REAL OpenAI API and returns valid content.
    This test makes a real network request.
    """
    if ChatService is None:
        pytest.fail("ChatService not found")

    service = ChatService(repo=repo, settings=settings)

    # We use a simple prompt to verify connectivity
    response = service.get_response("Hello, simply say 'Hello World'")

    assert isinstance(response, str)
    assert len(response) > 0


@pytest.mark.integration
def test_memory_recall_real_llm(settings, repo):
    """
    Test that the AI remembers information provided in previous turns.
    We simulate a 'restart' by creating a new ChatService instance
    connected to the same repository.
    """
    if ChatService is None:
        pytest.fail("ChatService not found")

    # Session 1: Teach the AI
    service1 = ChatService(repo=repo, settings=settings)
    service1.get_response("My name is Alice. Please remember this.")

    # Session 2: Verify Memory
    # We instantiate a NEW service, but pass the EXISTING repo (simulating persistent DB)
    service2 = ChatService(repo=repo, settings=settings)
    response = service2.get_response("What is my name?")

    # Assert the AI mentions "Alice"
    assert "Alice" in response
