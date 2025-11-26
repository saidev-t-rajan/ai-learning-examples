from app.agents.models import TripItinerary
from app.agents.planning import extract_budget_constraint, validate_itinerary


def test_budget_constraint_extraction_various_formats() -> None:
    """Verify budget extraction handles multiple natural language formats."""
    test_cases = [
        ("Plan trip for under NZ$500", 500.0),
        ("Budget of $1,200", 1200.0),
        ("for under $999", 999.0),
        ("Trip to Auckland for NZ$750", 750.0),
        ("No budget info here", None),
    ]

    for request, expected in test_cases:
        assert extract_budget_constraint(request) == expected


def test_itinerary_validation_accepts_valid_plan() -> None:
    """Verify validation passes for conforming itineraries."""
    itinerary = TripItinerary(
        destination="Auckland",
        origin="Wellington",
        duration_days=2,
        total_cost_nzd=450.0,
        flights=[],
        accommodation=[],
        activities=[],
        weather_summary="Good",
    )

    is_valid, _ = validate_itinerary(itinerary.model_dump(), max_budget=500.0)
    assert is_valid


def test_itinerary_validation_rejects_budget_violation() -> None:
    """Verify validation fails for over-budget plans."""
    itinerary = TripItinerary(
        destination="Auckland",
        origin="Wellington",
        duration_days=2,
        total_cost_nzd=550.0,
        flights=[],
        accommodation=[],
        activities=[],
        weather_summary="Good",
    )

    is_valid, message = validate_itinerary(itinerary.model_dump(), max_budget=500.0)
    assert not is_valid
    assert "550" in message
    assert "500" in message


def test_itinerary_validation_handles_invalid_json() -> None:
    """Verify validation gracefully handles malformed data."""
    invalid_dict = {"missing": "required_fields"}

    is_valid, message = validate_itinerary(invalid_dict, max_budget=500.0)
    assert not is_valid
    assert "Invalid" in message
