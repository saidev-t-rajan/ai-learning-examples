import pytest

from app.agents.planning import (
    extract_budget_constraint,
    validate_itinerary,
)
from app.agents.models import TripItinerary
from app.core.utils import extract_json_from_text


def test_extract_budget_constraint_from_natural_language():
    """Verify budget extraction from various phrasings."""
    assert extract_budget_constraint("Plan a trip for under NZ$500") == 500.0
    assert extract_budget_constraint("Trip with budget of $1000") == 1000.0
    assert extract_budget_constraint("for under $1,500") == 1500.0
    assert extract_budget_constraint("No budget mentioned") is None


def test_extract_json_from_markdown_code_block():
    """Verify JSON extraction from markdown code blocks."""
    text = '```json\n{"key": "value"}\n```'
    result = extract_json_from_text(text)
    assert result == {"key": "value"}


def test_extract_json_from_plain_text():
    """Verify JSON extraction from plain text."""
    text = 'Here is the plan: {"destination": "Auckland"}'
    result = extract_json_from_text(text)
    assert result == {"destination": "Auckland"}


def test_validate_itinerary_respects_budget():
    """Verify itinerary validation accepts within-budget plans."""
    itinerary = TripItinerary(
        destination="Auckland",
        origin="Wellington",
        duration_days=2,
        total_cost_nzd=450.0,
    )

    is_valid, msg = validate_itinerary(itinerary.model_dump(), max_budget=500.0)
    assert is_valid is True


def test_validate_itinerary_rejects_over_budget():
    """Verify itinerary validation rejects over-budget plans."""
    itinerary = TripItinerary(
        destination="Auckland",
        origin="Wellington",
        duration_days=2,
        total_cost_nzd=600.0,
    )

    is_valid, msg = validate_itinerary(itinerary.model_dump(), max_budget=500.0)
    assert is_valid is False
    assert "Budget exceeded" in msg


@pytest.mark.skip(reason="Requires complex mocking of OpenAI streaming")
def test_planning_service_yields_agent_steps():
    """Verify service yields step-by-step reasoning."""
    pass
