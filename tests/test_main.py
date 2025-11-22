from unittest.mock import patch
from io import StringIO
from app import main


def test_cli_exit_command():
    """
    Test that the CLI loop exits cleanly when the user types '/exit'.
    """
    # Simulate user inputs: First "Hello", then "/exit"
    user_inputs = ["Hello", "/exit"]

    # Mock input() to return our list, and stdout to capture prints
    with patch("builtins.input", side_effect=user_inputs):
        with patch("sys.stdout", new=StringIO()) as fake_out:
            main.start_chat()

            output = fake_out.getvalue()

            # Check for key UI elements
            assert "Datacom AI Assessment" in output
            assert "Goodbye" in output
