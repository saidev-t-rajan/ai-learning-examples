from app.core.utils import calculate_cost


def test_calculate_cost_gpt4o():
    # Rates: $2.50/1M input, $10.00/1M output
    # 1M input = $2.50
    # 1M output = $10.00
    cost = calculate_cost("gpt-4o", 1_000_000, 1_000_000)
    assert cost == 12.50


def test_calculate_cost_gpt35():
    # Rates: $0.50/1M input, $1.50/1M output (approx for turbo)
    # Just verifying it handles different models
    cost = calculate_cost("gpt-3.5-turbo", 1_000_000, 1_000_000)
    assert cost > 0


def test_calculate_cost_unknown():
    # Should probably return 0 or raise error, sticking to 0 for safety
    cost = calculate_cost("unknown-model", 100, 100)
    assert cost == 0.0


def test_calculate_cost_variations():
    # Test that "Gpt4o" (missing hyphen, mixed case) is handled as "gpt-4o"
    # Rates: $2.50/1M input, $10.00/1M output -> 1M each = $12.50
    cost = calculate_cost("Gpt4o", 1_000_000, 1_000_000)
    assert cost == 12.50

    # Test "gpt4o" (lowercase, missing hyphen)
    cost = calculate_cost("gpt4o", 1_000_000, 1_000_000)
    assert cost == 12.50
