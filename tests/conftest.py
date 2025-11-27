import pytest
from app.core.config import Settings
from app.db.chat_repository import ChatRepository
from app.agents.planning import PlanningService


@pytest.fixture
def settings():
    return Settings(_env_file=".env.test")


@pytest.fixture
def repo():
    return ChatRepository(db_path=":memory:")


@pytest.fixture
def planning_service(settings):
    return PlanningService(settings=settings)
