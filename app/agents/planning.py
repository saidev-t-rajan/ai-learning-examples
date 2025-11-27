import json
import re
import time
from collections.abc import Generator, Callable, Iterable
from typing import cast, Any

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

from app.core.config import Settings
from app.core.utils import calculate_cost, extract_json_from_text
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

    def _build_initial_messages(
        self, user_request: str
    ) -> list[ChatCompletionMessageParam]:
        """Build initial conversation messages."""
        return [
            {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
            {"role": "user", "content": user_request},
        ]

    def _call_llm(self, messages: list[ChatCompletionMessageParam]) -> Any:
        """Call LLM with current messages and tool definitions."""
        return self.client.chat.completions.create(
            model=self.settings.MODEL_NAME,
            messages=messages,
            tools=cast(Iterable[ChatCompletionToolParam], ALL_TOOLS),
            tool_choice="auto",
        )

    def _update_token_metrics(self, response: Any, tracker: dict[str, int]) -> None:
        """Update token usage metrics from response."""
        if response.usage:
            tracker["input_tokens"] += response.usage.prompt_tokens
            tracker["output_tokens"] += response.usage.completion_tokens

    def _execute_tool(
        self, tool_call: Any
    ) -> Generator[tuple[AgentStep, ChatCompletionMessageParam | None], None, None]:
        """Execute a single tool call and yield steps and message."""
        function_name = tool_call.function.name
        function_args_json = tool_call.function.arguments

        yield (
            AgentStep(
                step_type="tool_call", content=f"{function_name}({function_args_json})"
            ),
            None,
        )

        function_args = json.loads(function_args_json)
        executor = TOOL_EXECUTORS.get(function_name)

        if executor:
            safe_executor = cast(Callable[..., str], executor)
            tool_result = safe_executor(**function_args)
        else:
            tool_result = json.dumps({"error": f"Unknown tool: {function_name}"})

        yield AgentStep(step_type="tool_result", content=tool_result), None

        # Return message to append
        message: ChatCompletionMessageParam = {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": tool_result,
        }
        yield AgentStep(step_type="tool_result", content=""), message

    def _validate_and_yield_final_answer(
        self, content: str, max_budget: float | None
    ) -> Generator[AgentStep, None, None]:
        """Validate itinerary and yield final answer or validation error."""
        itinerary_dict = extract_json_from_text(content)

        if itinerary_dict and max_budget:
            is_valid, message = validate_itinerary(itinerary_dict, max_budget)
            if not is_valid:
                yield AgentStep(
                    step_type="validation_error",
                    content=f"Constraint violation: {message}",
                )

        yield AgentStep(step_type="final_answer", content=content)

    def _generate_metrics(
        self, start_time: float, total_input_tokens: int, total_output_tokens: int
    ) -> AgentStep:
        """Generate final metrics step."""
        end_time = time.time()
        cost = calculate_cost(
            self.settings.MODEL_NAME, total_input_tokens, total_output_tokens
        )

        return AgentStep(
            step_type="metrics",
            content=(
                f"prompt={total_input_tokens} "
                f"completion={total_output_tokens} "
                f"cost=${cost:.6f} "
                f"latency={int((end_time - start_time) * 1000)}ms"
            ),
        )

    def plan(self, user_request: str) -> Generator[AgentStep, None, None]:
        """Execute ReAct loop with tool calling until final itinerary or max iterations."""
        max_budget = extract_budget_constraint(user_request)
        messages = self._build_initial_messages(user_request)

        metrics_tracker = {"input_tokens": 0, "output_tokens": 0}
        start_time = time.time()
        max_iterations = 10

        for iteration in range(1, max_iterations + 1):
            response = self._call_llm(messages)
            self._update_token_metrics(response, metrics_tracker)

            assistant_message = response.choices[0].message

            if assistant_message.content:
                yield AgentStep(step_type="thought", content=assistant_message.content)

            if not assistant_message.tool_calls:
                # No more tools - yield final answer
                yield from self._validate_and_yield_final_answer(
                    assistant_message.content or "", max_budget
                )
                break

            # Add assistant message with tool calls
            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": cast(Any, assistant_message.tool_calls),
                }
            )

            # Execute all tool calls
            for tool_call in assistant_message.tool_calls:
                if tool_call.type != "function":
                    continue

                for step, message in self._execute_tool(tool_call):
                    if step.content:  # Don't yield empty steps
                        yield step
                    if message:
                        messages.append(message)

        # Generate final metrics
        yield self._generate_metrics(
            start_time,
            metrics_tracker["input_tokens"],
            metrics_tracker["output_tokens"],
        )


def extract_budget_constraint(user_request: str) -> float | None:
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


def validate_itinerary(
    itinerary_dict: dict[str, Any], max_budget: float | None
) -> tuple[bool, str]:
    try:
        itinerary = TripItinerary(**itinerary_dict)
    except Exception as e:
        return False, f"Invalid itinerary format: {e}"

    if max_budget and not itinerary.is_within_budget(max_budget):
        return False, (f"Budget exceeded: ${itinerary.total_cost_nzd} > ${max_budget}")

    return True, "Valid"
