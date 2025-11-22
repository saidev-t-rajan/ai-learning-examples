import os
import pytest


def test_rag_module_exists():
    """Verify that the app.rag module can be imported."""
    try:
        import app.rag  # noqa: F401
    except ImportError as e:
        pytest.fail(f"Could not import app.rag: {e}")


def test_agents_module_exists():
    """Verify that the app.agents module can be imported."""
    try:
        import app.agents  # noqa: F401
    except ImportError as e:
        pytest.fail(f"Could not import app.agents: {e}")


def test_sentence_transformers_installed():
    """Verify that sentence-transformers is installed and importable."""
    try:
        import sentence_transformers  # noqa: F401
    except ImportError as e:
        pytest.fail(f"Could not import sentence_transformers: {e}")


def test_config_loads():
    """Verify that the settings can be instantiated."""
    from app.core.config import Settings

    # We need to ensure required env vars are present or mock them.
    # Since we are running in a test environment, we might rely on .env.test or defaults.
    # If .env is missing, this might fail if we don't mock.
    # Let's mock the env vars for safety.

    os.environ["OPENAI_API_KEY"] = "sk-test-key"
    os.environ["OPENAI_BASE_URL"] = "https://api.openai.com/v1"

    try:
        settings = Settings()
        assert settings.OPENAI_API_KEY == "sk-test-key"
    except Exception as e:
        pytest.fail(f"Could not instantiate Settings: {e}")
