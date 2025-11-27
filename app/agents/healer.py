import json
import shutil
from collections.abc import Generator
from pathlib import Path

from app.agents.executor import CodeExecutor
from app.agents.models import HealerChunk, HealerMetrics, HealerExecutionStep
from app.agents.prompts import load_prompt_template
from app.core.chat_service import ChatService
from app.core.utils import extract_json_from_text


class HealerService:
    def __init__(
        self,
        chat_service: ChatService,
        work_dir: Path | str | None = None,
        max_attempts: int = 3,
        timeout_seconds: int = 30,
    ):
        self.chat_service = chat_service
        self.executor = CodeExecutor(work_dir=work_dir, timeout_seconds=timeout_seconds)
        self.max_attempts = max_attempts

    def _parse_plan(self, response_text: str) -> dict:
        """Parse LLM response into execution plan. Raises ValueError on failure."""
        plan = extract_json_from_text(response_text)
        if not plan:
            raise ValueError("No JSON found in response")

        files = plan.get("files", [])
        command = plan.get("command", "")

        if not files or not command:
            raise ValueError("Missing 'files' or 'command' in JSON response")

        return {"files": files, "command": command}

    def _execute_and_report(
        self, plan: dict, attempt_number: int, total_execution_time: float
    ) -> Generator[HealerExecutionStep, None, None]:
        """
        Execute plan and yield status chunks.
        """
        files = plan["files"]
        command = plan["command"]

        # Display generated code
        if len(files) == 1:
            yield HealerExecutionStep(
                chunk=HealerChunk(content=f"Generated code:\n{files[0]['content']}\n")
            )
        else:
            yield HealerExecutionStep(
                chunk=HealerChunk(
                    content=f"Generated {len(files)} files. Command: {command}\n"
                )
            )

        # Execute
        result = self.executor.execute_project(files, command)
        total_execution_time += result.execution_time_seconds

        if result.success:
            yield HealerExecutionStep(
                chunk=HealerChunk(content="\n✓ All tests passed!\n")
            )
            yield HealerExecutionStep(
                chunk=HealerChunk(
                    metrics=HealerMetrics(
                        total_attempts=attempt_number,
                        successful=True,
                        total_execution_time_seconds=total_execution_time,
                        final_exit_code=result.exit_code,
                    )
                ),
                is_success=True,
                execution_time=result.execution_time_seconds,
            )
        else:
            yield HealerExecutionStep(chunk=HealerChunk(content="\n✗ Tests failed\n"))
            yield HealerExecutionStep(
                chunk=HealerChunk(content=f"Error output:\n{result.stderr}\n"),
                is_success=False,
                execution_time=result.execution_time_seconds,
            )

    def heal_code(self, task_description: str) -> Generator[HealerChunk, None, None]:
        """Generate and iteratively fix code for a task."""
        available_tools = self._discover_tools()
        system_prompt = self._build_system_prompt(available_tools)
        current_prompt = task_description
        total_execution_time = 0.0

        for attempt_number in range(1, self.max_attempts + 1):
            yield HealerChunk(
                content=f"\n=== Attempt {attempt_number}/{self.max_attempts} ===\n"
            )

            response_text = self._get_llm_response(system_prompt, current_prompt)

            try:
                plan = self._parse_plan(response_text)
            except (json.JSONDecodeError, ValueError) as e:
                yield HealerChunk(content=f"Failed to parse plan: {e}\n")
                if attempt_number < self.max_attempts:
                    current_prompt = f"Your previous response was invalid JSON. Error: {e}. Return ONLY valid JSON."
                    continue
                break

            # Execute and report results
            for step in self._execute_and_report(
                plan, attempt_number, total_execution_time
            ):
                if step.chunk:
                    yield step.chunk
                total_execution_time += step.execution_time
                if step.is_success:
                    return

            # Prepare fix prompt for next attempt
            if attempt_number < self.max_attempts:
                current_prompt = self._build_fix_prompt(
                    json.dumps(plan, indent=2), "See error output above"
                )

        # Max attempts reached
        yield HealerChunk(
            metrics=HealerMetrics(
                total_attempts=self.max_attempts,
                successful=False,
                total_execution_time_seconds=total_execution_time,
                final_exit_code=-1,
            )
        )

    def _discover_tools(self) -> list[str]:
        """Check which compilers/runtimes are available on PATH."""
        candidates = ["python", "python3", "cargo", "go", "node", "npm", "gcc", "javac"]
        found = []
        for tool in candidates:
            if shutil.which(tool):
                found.append(tool)
        return found

    def _get_llm_response(self, system_prompt: str, user_prompt: str) -> str:
        full_response = ""
        for chunk in self.chat_service.get_response(
            message=user_prompt, system_message=system_prompt
        ):
            if chunk.content:
                full_response += chunk.content
        return full_response

    def _build_system_prompt(self, available_tools: list[str]) -> str:
        tools_str = ", ".join(available_tools)
        template = load_prompt_template("healer_system")
        return template.format(tools=tools_str)

    def _build_fix_prompt(self, previous_plan_json: str, error_message: str) -> str:
        template = load_prompt_template("healer_fix")
        return template.format(
            error_message=error_message, previous_plan=previous_plan_json
        )
