PRICING: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
}


def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate the cost of a request based on the model and token usage.
    Rates are per 1,000,000 tokens.
    """
    model_lower = model_name.lower()

    if "gpt-4o-mini" in model_lower:
        rates = PRICING["gpt-4o-mini"]
    elif "gpt-4o" in model_lower or "gpt4o" in model_lower:
        rates = PRICING["gpt-4o"]
    elif "gpt-3.5" in model_lower:
        rates = PRICING["gpt-3.5-turbo"]
    else:
        return 0.0

    input_cost = (input_tokens / 1_000_000) * rates["input"]
    output_cost = (output_tokens / 1_000_000) * rates["output"]

    return input_cost + output_cost
