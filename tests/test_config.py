import pytest
import os
from pydantic import ValidationError
from app.core.config import Settings


def test_settings_validation():
    """
    Test that Settings validation fails if required variables are missing.

    Teaching Moment:
    Pydantic Settings automatically validates environment variables.
    We ensure the app crashes early if 'OPENAI_API_KEY' is missing.
    """
    # Manual environment management instead of patch.dict
    original_key = os.environ.get("OPENAI_API_KEY")

    try:
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

        # Assert that creating Settings without the key raises a validation error
        with pytest.raises(ValidationError):
            Settings(_env_file=None)

    finally:
        # Restore environment
        if original_key is not None:
            os.environ["OPENAI_API_KEY"] = original_key


def test_settings_loading():
    """
    Test that Settings loads correctly from the real environment (.env.test).
    """
    # The pytest-dotenv plugin (configured in pytest.ini) loads .env.test into os.environ
    # So Settings() will naturally pick up those values.
    config = Settings()

    # Verify against the actual values in .env.test
    assert config.MODEL_NAME == "Gpt4o"
    assert config.OPENAI_API_KEY == os.environ["OPENAI_API_KEY"]
    assert "aiunifier" in config.OPENAI_BASE_URL
