import pytest
from app.core.config import Settings


@pytest.fixture
def settings():
    return Settings(_env_file=".env.test")
