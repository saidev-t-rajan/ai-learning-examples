import time
import logging
import functools

logger = logging.getLogger(__name__)


def time_execution(threshold: float = 0.3):
    """
    Decorator to measure execution time of a function.
    Logs a warning if execution time exceeds the threshold (seconds).
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            duration = end_time - start_time

            if duration > threshold:
                logger.warning(
                    f"Execution of '{func.__name__}' took {duration:.2f}s (>{threshold:.2f}s)"
                )
            else:
                logger.debug(f"Execution of '{func.__name__}' took {duration:.2f}s")

            return result

        return wrapper

    return decorator


def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate the cost of a request based on the model and token usage.
    Rates are per 1,000,000 tokens.
    """
    # Pricing per 1M tokens (approximate as of late 2024/2025 context)
    PRICING = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    }

    model_key = model_name.lower()
    # Handle model snapshots or variations (e.g. gpt-4o-2024-08-06)
    if "gpt-4o-mini" in model_key:
        model_key = "gpt-4o-mini"
    elif "gpt-4o" in model_key or "gpt4o" in model_key:
        model_key = "gpt-4o"
    elif "gpt-3.5-turbo" in model_key:
        model_key = "gpt-3.5-turbo"

    if model_key not in PRICING:
        return 0.0

    rates = PRICING[model_key]
    input_cost = (input_tokens / 1_000_000) * rates["input"]
    output_cost = (output_tokens / 1_000_000) * rates["output"]

    return input_cost + output_cost
