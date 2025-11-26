from unittest.mock import Mock
from app.cli import CLI, PLAN
from app.core.config import Settings
from app.agents.models import AgentStep


def test_handle_plan_command_routes_to_planning_service():
    """Verify /plan command routes to planning service."""
    mock_chat_service = Mock()
    mock_rag_service = Mock()
    mock_planning_service = Mock()
    settings = Settings()

    cli = CLI(
        chat_service=mock_chat_service,
        rag_service=mock_rag_service,
        settings=settings,
        planning_service=mock_planning_service,
    )

    # Configure mock to return empty iterator to avoid TypeError in CLI loop
    mock_planning_service.plan.return_value = []

    cli._handle_command(f"{PLAN} Plan a trip to Auckland")

    mock_planning_service.plan.assert_called_once()


def test_handle_plan_command_displays_formatted_output(capsys):
    """Verify /plan command shows formatted agent steps."""
    mock_service = Mock()
    mock_service.plan.return_value = [
        AgentStep(step_type="thought", content="Checking flights"),
        AgentStep(step_type="tool_call", content="get_flight_prices(...)"),
        AgentStep(step_type="final_answer", content='{"destination": "Auckland"}'),
    ]

    cli = CLI(Mock(), Mock(), Settings(), planning_service=mock_service)
    cli._handle_plan(f"{PLAN} trip to Auckland")

    captured = capsys.readouterr()
    assert "[Thinking]" in captured.out
    assert "[Tool Call]" in captured.out
    assert "[Final Plan]" in captured.out


def test_handle_plan_command_when_service_not_available(capsys):
    """Verify graceful handling when planning service is None."""
    cli = CLI(Mock(), Mock(), Settings(), planning_service=None)
    cli._handle_plan(f"{PLAN} trip to Auckland")

    captured = capsys.readouterr()
    assert "not available" in captured.out
