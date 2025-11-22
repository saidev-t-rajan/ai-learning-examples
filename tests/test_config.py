import pytest
import os
from unittest.mock import patch
from pydantic import ValidationError

# TDD: We expect this import to fail initially
try:
    from app.core.config import Settings
except ImportError:
    Settings = None


def test_settings_validation():
    """
    Test that Settings validation fails if required variables are missing.

    Teaching Moment:
    Pydantic Settings automatically validates environment variables.
    We ensure the app crashes early if 'OPENAI_API_KEY' is missing.
    """
    if Settings is None:
        pytest.fail("app.core.config module not found")

    # Clear the variable if it exists to test failure
    # We use patch.dict to ensure the deletion is temporary
    with patch.dict(os.environ):
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

        # Assert that creating Settings without the key raises a validation error
        with pytest.raises(ValidationError):
            Settings(_env_file=None)


def test_settings_loading():
    """
    Test that Settings loads correctly when variables are present.
    """
    if Settings is None:
        pytest.fail("app.core.config module not found")

    # Mock the environment variables safely using patch.dict
    mock_env = {
        "OPENAI_API_KEY": "sk-test-key",
        "OPENAI_BASE_URL": "https://mock.api/v1",
        "MODEL_NAME": "gpt-4o-mini",
    }

    with patch.dict(os.environ, mock_env):
        # Reload settings or instantiate new one
        config = Settings()

        assert config.OPENAI_API_KEY == "sk-test-key"
        assert config.OPENAI_BASE_URL == "https://mock.api/v1"
        assert config.MODEL_NAME == "gpt-4o-mini"
