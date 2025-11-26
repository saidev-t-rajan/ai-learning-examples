import json
from app.agents.tools import execute_get_flight_prices, execute_get_weather_forecast


def test_get_flight_prices_returns_valid_json():
    """Verify flight tool returns parseable JSON with flight data."""
    result_json = execute_get_flight_prices(destination="Auckland")
    result = json.loads(result_json)

    assert "flights" in result
    assert len(result["flights"]) > 0
    assert result["flights"][0]["destination"] == "Auckland"


def test_get_flight_prices_filters_by_max_price():
    """Verify flight tool respects max_price filter."""
    result_json = execute_get_flight_prices(destination="Auckland", max_price=160.0)
    result = json.loads(result_json)

    for flight in result["flights"]:
        assert flight["price_nzd"] <= 160.0


def test_get_flight_prices_returns_empty_for_unknown_destination():
    """Verify flight tool handles unknown destinations gracefully."""
    result_json = execute_get_flight_prices(destination="UnknownCity")
    result = json.loads(result_json)

    assert result["flights"] == []


def test_get_weather_forecast_returns_valid_json():
    """Verify weather tool returns parseable JSON with forecast data."""
    result_json = execute_get_weather_forecast(city="Auckland", days=2)
    result = json.loads(result_json)

    assert result["city"] == "Auckland"
    assert len(result["forecast"]) == 2
    assert "condition" in result["forecast"][0]


def test_get_weather_forecast_limits_days():
    """Verify weather tool respects days parameter."""
    result_json = execute_get_weather_forecast(city="Auckland", days=1)
    result = json.loads(result_json)

    assert len(result["forecast"]) == 1
