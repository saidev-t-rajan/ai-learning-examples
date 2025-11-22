import pytest
from app.db.memory import ChatRepository


@pytest.fixture
def repo():
    # Use in-memory DB for speed and isolation
    return ChatRepository(db_path=":memory:")


def test_migration_creates_table(repo):
    """
    Test that initializing the repo creates the correct table schema.
    """
    with repo._get_connection() as conn:
        # Check table existence
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chat_history'"
        )
        assert cur.fetchone() is not None

        # Check version
        cur.execute("PRAGMA user_version")
        assert cur.fetchone()[0] == 1


def test_add_and_get_messages(repo):
    """
    Test adding messages and retrieving them in the correct order.
    """
    # Add messages
    repo.add_message("user", "Hello")
    repo.add_message("assistant", "Hi there")
    repo.add_message("user", "How are you?")

    # Retrieve messages
    messages = repo.get_recent_messages(limit=10)

    assert len(messages) == 3
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "Hi there"
    assert messages[2]["role"] == "user"
    assert messages[2]["content"] == "How are you?"


def test_get_recent_messages_limit(repo):
    """
    Test that the limit parameter works correctly.
    """
    for i in range(5):
        repo.add_message("user", f"Message {i}")

    # Get only last 2
    messages = repo.get_recent_messages(limit=2)

    assert len(messages) == 2
    # Should be the last two: Message 3 and Message 4
    assert messages[0]["content"] == "Message 3"
    assert messages[1]["content"] == "Message 4"
