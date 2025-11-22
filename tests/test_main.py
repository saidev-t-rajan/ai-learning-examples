from unittest.mock import patch
from io import StringIO

# TDD: Import will fail initially
try:
    from app import main
except ImportError:
    main = None


def test_cli_exit_command():
    """
    Test that the CLI loop exits cleanly when the user types '/exit'.
    """
    user_inputs = ["Hello", "/exit"]

    with patch("builtins.input", side_effect=user_inputs):
        with patch("sys.stdout", new=StringIO()) as fake_out:
            main.start_chat()
            output = fake_out.getvalue()
            assert "Datacom AI Assessment" in output
            assert "Goodbye" in output


def test_cli_uses_service():
    """
    Test that the CLI actually calls ChatService to get a response.

    Teaching Moment:
    We mock the 'ChatService' class. We don't care *what* it returns
    (that's tested in test_chat_service.py). We only care that main.py
    *asks* for a response. This is "Integration Testing" the wiring.
    """
    user_inputs = ["Hello AI", "/exit"]

    # We mock the CLASS itself, so when main.py calls ChatService(), it gets our mock
    with patch("app.main.ChatService") as MockServiceClass:
        # Setup the mock instance
        mock_instance = MockServiceClass.return_value
        mock_instance.get_response.return_value = "Mocked Response"

        with patch("builtins.input", side_effect=user_inputs):
            with patch("sys.stdout", new=StringIO()) as fake_out:
                main.start_chat()

                # Verify our mock was called with the user's input
                mock_instance.get_response.assert_called_once_with("Hello AI")

                # Verify the CLI printed the mocked response
                assert "Mocked Response" in fake_out.getvalue()
