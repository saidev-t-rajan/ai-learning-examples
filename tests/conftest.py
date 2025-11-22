import pytest
from app.core.config import Settings
from app.db.memory import ChatRepository


@pytest.fixture
def settings():
    return Settings(_env_file=".env.test")


@pytest.fixture
def repo():
    return ChatRepository(db_path=":memory:")
