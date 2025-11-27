import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast


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
    model_lower = model_name.lower()

    # Match patterns to model keys
    for pattern, key in [
        ("gpt-4o-mini", "gpt-4o-mini"),
        ("gpt-4o", "gpt-4o"),
        ("gpt4o", "gpt-4o"),
        ("gpt-3.5", "gpt-3.5-turbo"),
    ]:
        if pattern in model_lower:
            pricing = MODEL_PRICING[key]
            return (input_tokens / 1_000_000) * pricing.input_rate + (
                output_tokens / 1_000_000
            ) * pricing.output_rate

    return 0.0


class ValidationError(Exception):
    pass


def validate_file_path(
    path_str: str, allowed_extensions: list[str] | None = None
) -> Path:
    try:
        path = Path(path_str).resolve()
    except Exception as e:
        raise ValidationError(f"Invalid path format: {path_str}") from e

    if not path.exists():
        raise ValidationError(f"File not found: {path_str}")

    if not path.is_file():
        raise ValidationError(f"Path is not a file: {path_str}")

    if allowed_extensions:
        suffixes = [ext.lower() for ext in allowed_extensions]
        if path.suffix.lower() not in suffixes:
            raise ValidationError(
                f"Invalid file type: {path.name}. Allowed: {', '.join(allowed_extensions)}"
            )

    return path


def validate_directory_path(path_str: str) -> Path:
    try:
        path = Path(path_str).resolve()
    except Exception as e:
        raise ValidationError(f"Invalid path format: {path_str}") from e

    if not path.exists():
        raise ValidationError(f"Directory not found: {path_str}")

    if not path.is_dir():
        raise ValidationError(f"Path is not a directory: {path_str}")

    return path


def extract_json_from_text(text: str) -> dict[str, Any] | None:
    """Extract first JSON object from text, checking markdown blocks first."""
    markdown_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
    match = re.search(markdown_pattern, text, re.DOTALL)

    if match:
        json_str = match.group(1)
    else:
        brace_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
        match = re.search(brace_pattern, text, re.DOTALL)
        if not match:
            return None
        json_str = match.group(0)

    try:
        parsed = json.loads(json_str)
        return cast(dict[str, Any], parsed) if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None
