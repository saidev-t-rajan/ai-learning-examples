import pytest
from unittest.mock import patch
from io import StringIO

# TDD: Import will fail initially
try:
    from app import main
except ImportError:
    main = None


@pytest.mark.integration
def test_cli_exit_command():
    """
    Test that the CLI loop exits cleanly when the user types '/exit'.
    """
    user_inputs = ["/exit"]

    with patch("builtins.input", side_effect=user_inputs):
        with patch("sys.stdout", new=StringIO()) as fake_out:
            main.start_chat()
            output = fake_out.getvalue()
            assert "Datacom AI Assessment" in output
            assert "Goodbye" in output


@pytest.mark.integration
def test_cli_uses_service():
    """
    Test that the CLI actually calls the REAL ChatService.
    """
    user_inputs = ["Hello AI", "/exit"]

    with patch("builtins.input", side_effect=user_inputs):
        with patch("sys.stdout", new=StringIO()) as fake_out:
            main.start_chat()

            output = fake_out.getvalue()

            # Verify the CLI printed the AI prefix
            assert "AI: " in output
            # We can't assert the exact text, but we know it shouldn't be empty
            assert len(output.strip()) > 0
