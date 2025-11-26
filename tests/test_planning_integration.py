import pytest
from app.agents.planning import PlanningService


@pytest.mark.integration
def test_planning_agent_end_to_end(settings):
    """Test full planning workflow with real OpenAI API."""
    service = PlanningService(settings=settings)

    steps = list(service.plan("Plan a 2-day trip to Auckland for under NZ$500"))

    thoughts = [s for s in steps if s.step_type == "thought"]
    assert len(thoughts) > 0

    tool_calls = [s for s in steps if s.step_type == "tool_call"]
    assert len(tool_calls) >= 2

    final = [s for s in steps if s.step_type == "final_answer"]
    assert len(final) == 1

    metrics = [s for s in steps if s.step_type == "metrics"]
    assert len(metrics) == 1
