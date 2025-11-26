from dataclasses import dataclass


@dataclass
class ModelPricing:
    input_rate: float
    output_rate: float


# Pricing per 1M tokens
MODEL_PRICING = {
    "gpt-4o-mini": ModelPricing(0.15, 0.60),
    "gpt-4o": ModelPricing(2.50, 10.00),
    "gpt-3.5-turbo": ModelPricing(0.50, 1.50),
}


def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate the cost of a request based on the model and token usage.
    Rates are per 1,000,000 tokens.
    """
    model_lower = model_name.lower()

    # Order matters: check more specific patterns first
    check_order = [
        ("gpt-4o-mini", "gpt-4o-mini"),
        ("gpt-4o", "gpt-4o"),
        ("gpt4o", "gpt-4o"),
        ("gpt-3.5", "gpt-3.5-turbo"),
    ]

    pricing = None
    for substring, key in check_order:
        if substring in model_lower:
            pricing = MODEL_PRICING[key]
            break

    if not pricing:
        return 0.0

    input_cost = (input_tokens / 1_000_000) * pricing.input_rate
    output_cost = (output_tokens / 1_000_000) * pricing.output_rate

    return input_cost + output_cost
