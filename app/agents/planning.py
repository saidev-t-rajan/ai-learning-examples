import json
import re
import time
from collections.abc import Generator, Callable, Iterable
from typing import cast, Any

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

from app.core.config import Settings
from app.core.utils import calculate_cost
from app.agents.tools import ALL_TOOLS, TOOL_EXECUTORS
from app.agents.models import AgentStep, TripItinerary


PLANNING_SYSTEM_PROMPT = """You are a travel planning assistant. When given a trip request:

1. Think step-by-step about what information you need
2. Use available tools to gather flight prices and weather forecasts
3. Ensure your plan stays within the specified budget
4. Output a final itinerary in JSON format with these fields:
   - destination: string
   - origin: string
   - duration_days: integer
   - total_cost_nzd: float
   - flights: list of flight objects
   - weather_summary: string

Before calling tools, explain your reasoning clearly."""


class PlanningService:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.client = OpenAI(
            api_key=self.settings.OPENAI_API_KEY,
            base_url=self.settings.OPENAI_BASE_URL,
        )

    def plan(self, user_request: str) -> Generator[AgentStep, None, None]:
        """
        Execute ReAct loop: Reasoning → Action → Observation.
        Yields AgentStep objects showing agent thought process.
        """
        max_budget = extract_budget_constraint(user_request)

        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
            {"role": "user", "content": user_request},
        ]

        max_iterations = 10
        iteration = 0
        start_time = time.time()
        total_input_tokens = 0
        total_output_tokens = 0

        while iteration < max_iterations:
            iteration += 1

            response = self.client.chat.completions.create(
                model=self.settings.MODEL_NAME,
                messages=messages,
                tools=cast(Iterable[ChatCompletionToolParam], ALL_TOOLS),
                tool_choice="auto",
            )

            if response.usage:
                total_input_tokens += response.usage.prompt_tokens
                total_output_tokens += response.usage.completion_tokens

            assistant_message = response.choices[0].message

            if assistant_message.content:
                yield AgentStep(step_type="thought", content=assistant_message.content)

            if not assistant_message.tool_calls:
                final_content = assistant_message.content or ""

                itinerary_dict = extract_json_from_response(final_content)
                if itinerary_dict and max_budget:
                    is_valid, message = validate_itinerary(itinerary_dict, max_budget)
                    if not is_valid:
                        yield AgentStep(
                            step_type="validation_error",
                            content=f"Constraint violation: {message}",
                        )

                yield AgentStep(step_type="final_answer", content=final_content)
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": cast(Any, assistant_message.tool_calls),
                }
            )

            for tool_call in assistant_message.tool_calls:
                if tool_call.type != "function":
                    continue

                function_name = tool_call.function.name
                function_args_json = tool_call.function.arguments

                yield AgentStep(
                    step_type="tool_call",
                    content=f"{function_name}({function_args_json})",
                )

                function_args = json.loads(function_args_json)
                executor = TOOL_EXECUTORS.get(function_name)

                if executor:
                    # Cast to Callable to satisfy Mypy
                    safe_executor = cast(Callable[..., str], executor)
                    tool_result = safe_executor(**function_args)
                else:
                    tool_result = json.dumps(
                        {"error": f"Unknown tool: {function_name}"}
                    )

                yield AgentStep(step_type="tool_result", content=tool_result)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result,
                    }
                )

        end_time = time.time()
        cost = calculate_cost(
            self.settings.MODEL_NAME, total_input_tokens, total_output_tokens
        )

        yield AgentStep(
            step_type="metrics",
            content=(
                f"prompt={total_input_tokens} "
                f"completion={total_output_tokens} "
                f"cost=${cost:.6f} "
                f"latency={int((end_time - start_time) * 1000)}ms"
            ),
        )


def extract_budget_constraint(user_request: str) -> float | None:
    """Extract budget from natural language patterns."""
    patterns = [
        r"under\s+(?:NZ)?\$?([\d,]+)",
        r"for\s+(?:NZ)?\$?([\d,]+)",
        r"budget\s+of\s+(?:NZ)?\$?([\d,]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, user_request, re.IGNORECASE)
        if match:
            amount_str = match.group(1).replace(",", "")
            return float(amount_str)

    return None


def extract_json_from_response(text: str) -> dict[str, Any] | None:
    """Extract JSON object from markdown code blocks or plain text."""
    json_block_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
    match = re.search(json_block_pattern, text, re.DOTALL)

    if match:
        json_str = match.group(1)
    else:
        json_obj_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
        match = re.search(json_obj_pattern, text, re.DOTALL)
        if match:
            json_str = match.group(0)
        else:
            return None

    try:
        return cast(dict[str, Any], json.loads(json_str))
    except json.JSONDecodeError:
        return None


def validate_itinerary(
    itinerary_dict: dict[str, Any], max_budget: float | None
) -> tuple[bool, str]:
    """Validate itinerary against constraints."""
    try:
        itinerary = TripItinerary(**itinerary_dict)
    except Exception as e:
        return False, f"Invalid itinerary format: {e}"

    if max_budget and not itinerary.is_within_budget(max_budget):
        return False, (f"Budget exceeded: ${itinerary.total_cost_nzd} > ${max_budget}")

    return True, "Valid"
