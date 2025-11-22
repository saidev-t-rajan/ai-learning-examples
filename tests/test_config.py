import pytest
import os
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
    if "OPENAI_API_KEY" in os.environ:
        del os.environ["OPENAI_API_KEY"]

    # Assert that creating Settings without the key raises a validation error
    with pytest.raises(ValidationError):
        Settings()


def test_settings_loading():
    """
    Test that Settings loads correctly when variables are present.
    """
    if Settings is None:
        pytest.fail("app.core.config module not found")

    # Mock the environment variable
    os.environ["OPENAI_API_KEY"] = "sk-test-key"
    os.environ["MODEL_NAME"] = "gpt-4o-mini"

    # Reload settings or instantiate new one
    config = Settings()

    assert config.OPENAI_API_KEY == "sk-test-key"
    assert config.MODEL_NAME == "gpt-4o-mini"
