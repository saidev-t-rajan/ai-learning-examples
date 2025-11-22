import pytest

# TDD: Import will fail initially
try:
    from app.core.chat_service import ChatService
except ImportError:
    ChatService = None


def test_chat_service_structure():
    """
    Test that the ChatService class exists and has a get_response method.

    Teaching Moment:
    We define the 'Interface' of our service before we write the implementation.
    This ensures our CLI code (the consumer) dictates the design.
    """
    if ChatService is None:
        pytest.fail("app.core.chat_service module not found")

    service = ChatService()
    assert hasattr(service, "get_response")


def test_get_response_basic():
    """
    Test that get_response returns a valid string.
    """
    if ChatService is None:
        pytest.fail("ChatService not found")

    service = ChatService()
    # We use a mock or simple logic for now, just testing the return type
    response = service.get_response("Hello AI")

    assert isinstance(response, str)
    assert len(response) > 0
