from pydantic import BaseModel


FLIGHT_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_flight_prices",
        "description": "Retrieve available flights with prices for a destination",
        "parameters": {
            "type": "object",
            "properties": {
                "destination": {
                    "type": "string",
                    "description": "Destination city (e.g., 'Auckland', 'Wellington')",
                },
                "origin": {
                    "type": "string",
                    "description": "Origin city",
                    "default": "Wellington",
                },
                "max_price": {
                    "type": "number",
                    "description": "Maximum acceptable price in NZD",
                },
            },
            "required": ["destination"],
        },
    },
}

WEATHER_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_weather_forecast",
        "description": "Get weather forecast for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "days": {
                    "type": "integer",
                    "description": "Number of days to forecast",
                    "default": 2,
                },
            },
            "required": ["city"],
        },
    },
}

ALL_TOOLS = [FLIGHT_TOOL_SCHEMA, WEATHER_TOOL_SCHEMA]


class FlightOption(BaseModel):
    airline: str
    origin: str
    destination: str
    departure_time: str
    arrival_time: str
    price_nzd: float


class FlightResult(BaseModel):
    flights: list[FlightOption]


class WeatherDay(BaseModel):
    date: str
    condition: str
    temp_high_c: int
    temp_low_c: int


class WeatherResult(BaseModel):
    city: str
    forecast: list[WeatherDay]


MOCK_FLIGHTS = {
    "Auckland": [
        FlightOption(
            airline="Air NZ",
            origin="Wellington",
            destination="Auckland",
            departure_time="2025-12-01T08:00:00",
            arrival_time="2025-12-01T09:15:00",
            price_nzd=180.00,
        ),
        FlightOption(
            airline="Jetstar",
            origin="Wellington",
            destination="Auckland",
            departure_time="2025-12-01T14:30:00",
            arrival_time="2025-12-01T15:45:00",
            price_nzd=150.00,
        ),
    ]
}

MOCK_WEATHER = {
    "Auckland": [
        WeatherDay(date="2025-12-01", condition="Sunny", temp_high_c=24, temp_low_c=18),
        WeatherDay(
            date="2025-12-02", condition="Partly Cloudy", temp_high_c=22, temp_low_c=17
        ),
    ]
}


def execute_get_flight_prices(
    destination: str, origin: str = "Wellington", max_price: float | None = None
) -> str:
    flights = MOCK_FLIGHTS.get(destination, [])

    if max_price:
        flights = [f for f in flights if f.price_nzd <= max_price]

    result = FlightResult(flights=flights)
    return result.model_dump_json()


def execute_get_weather_forecast(city: str, days: int = 2) -> str:
    forecast = MOCK_WEATHER.get(city, [])[:days]
    result = WeatherResult(city=city, forecast=forecast)
    return result.model_dump_json()


TOOL_EXECUTORS = {
    "get_flight_prices": execute_get_flight_prices,
    "get_weather_forecast": execute_get_weather_forecast,
}
