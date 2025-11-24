import pytest
import sys
import subprocess
from pathlib import Path

# Get the root directory of the project (where app/ is)
PROJECT_ROOT = Path(__file__).parent.parent


@pytest.mark.integration
def test_cli_exit_command_real():
    """
    Test that the CLI loop exits cleanly when the user types '/exit'.
    Runs the actual process.
    """
    # Run the module as a script
    result = subprocess.run(
        [sys.executable, "-m", "app.main"],
        input="/exit\n",
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=60,  # Safety timeout - increased for imports
    )

    assert result.returncode == 0
    assert "Datacom AI Assessment" in result.stdout
    assert "Goodbye" in result.stdout


@pytest.mark.integration
def test_cli_uses_service_real():
    """
    Test that the CLI actually calls the REAL ChatService.
    Runs the actual process.
    """
    # We send "Hello" then "/exit"
    # Note: This will hit the real OpenAI API and cost money/tokens.
    input_str = "Hello AI\n/exit\n"

    result = subprocess.run(
        [sys.executable, "-m", "app.main"],
        input=input_str,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=60,  # Longer timeout for API call
    )

    assert result.returncode == 0
    # Check for CLI prompt/prefixes
    assert (
        "You: " in result.stdout or "You:" in result.stdout
    )  # Depending on implementation
    assert "AI: " in result.stdout

    # We expect some response content (hard to predict exact text)
    # But checking for the AI prefix confirms the loop ran.
