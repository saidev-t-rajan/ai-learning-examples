import pytest
from app.core.chat_service import ChatService


def test_chat_service_structure(settings, repo):
    """
    Test that the ChatService class exists and has a get_response method.
    """
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
    # Consume generator
    list(service.get_response("Hi"))

    # 2. Check DB directly to verify persistence
    history = repo.get_recent_messages(limit=10)
    # Expect: User("Hi") + Assistant(Response)
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[0].content == "Hi"
    assert history[1].role == "assistant"
    assert len(history[1].content) > 0

    # 3. Send another message to verify context
    # We can't easily prove the model "remembered" without a complex prompt,
    # but we can verify the DB grew.
    list(service.get_response("Bye"))

    history_new = repo.get_recent_messages(limit=10)
    # Expect: User("Hi") + Assistant(Response1) + User("Bye") + Assistant(Response2)
    assert len(history_new) == 4
    assert history_new[2].role == "user"
    assert history_new[2].content == "Bye"


@pytest.mark.integration
def test_get_response_real_api(settings, repo):
    """
    Test that get_response calls the REAL OpenAI API and returns valid content.
    This test makes a real network request.
    """
    service = ChatService(repo=repo, settings=settings)

    # We use a simple prompt to verify connectivity
    # Consume generator
    chunks = list(service.get_response("Hello, simply say 'Hello World'"))

    # Last chunk should be metrics
    # Filter for chunks that have metrics
    metrics_chunks = [c for c in chunks if c.metrics]
    assert len(metrics_chunks) == 1
    metrics = metrics_chunks[0].metrics

    from app.core.models import ChatMetrics

    assert isinstance(metrics, ChatMetrics)

    # Reconstruct text
    response = "".join([c.content for c in chunks if c.content])

    assert isinstance(response, str)
    assert len(response) > 0


@pytest.mark.integration
def test_memory_recall_real_llm(settings, repo):
    """
    Test that the AI remembers information provided in previous turns.
    We simulate a 'restart' by creating a new ChatService instance
    connected to the same repository.
    """
    # Session 1: Teach the AI
    service1 = ChatService(repo=repo, settings=settings)
    list(service1.get_response("My name is Alice. Please remember this."))

    # Session 2: Verify Memory
    # We instantiate a NEW service, but pass the EXISTING repo (simulating persistent DB)
    service2 = ChatService(repo=repo, settings=settings)
    chunks = list(service2.get_response("What is my name?"))
    response = "".join([c.content for c in chunks if c.content])

    # Assert the AI mentions "Alice"
    assert "Alice" in response
