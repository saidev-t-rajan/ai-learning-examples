import pytest

# TDD: Import will fail initially
try:
    from app.core.chat_service import ChatService
except ImportError:
    ChatService = None


def test_chat_service_structure():
    """
    Test that the ChatService class exists and has a get_response method.
    """
    if ChatService is None:
        pytest.fail("app.core.chat_service module not found")

    service = ChatService()
    assert hasattr(service, "get_response")


@pytest.mark.integration
def test_get_response_real_api():
    """
    Test that get_response calls the REAL OpenAI API and returns valid content.
    This test makes a real network request.
    """
    if ChatService is None:
        pytest.fail("ChatService not found")

    service = ChatService()

    # We use a simple prompt to verify connectivity
    response = service.get_response("Hello, simply say 'Hello World'")

    assert isinstance(response, str)
    assert len(response) > 0
    # The model might add punctuation, so we check if the core phrase is present roughly or just length
    # Given it's a real LLM, we can't be 100% deterministic on the output string, but checking it's a non-empty string is good for connectivity.
