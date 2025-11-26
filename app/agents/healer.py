import json
import re
import shutil
from collections.abc import Generator
from pathlib import Path
from typing import cast, Any

from app.agents.executor import CodeExecutor
from app.agents.models import HealerChunk, HealerMetrics
from app.core.chat_service import ChatService


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

    def heal_code(self, task_description: str) -> Generator[HealerChunk, None, None]:
        available_tools = self._discover_tools()
        system_prompt = self._build_system_prompt(available_tools)
        current_prompt = task_description

        total_execution_time = 0.0

        for attempt_number in range(1, self.max_attempts + 1):
            yield HealerChunk(
                content=f"\n=== Attempt {attempt_number}/{self.max_attempts} ===\n"
            )

            response_text = self._get_llm_response(system_prompt, current_prompt)

            # Try to parse JSON from response
            try:
                plan = self._extract_json_plan(response_text)
                files = plan.get("files", [])
                command = plan.get("command", "")

                # Basic validation
                if not files or not command:
                    raise ValueError("Missing 'files' or 'command' in JSON response")

            except (json.JSONDecodeError, ValueError) as e:
                yield HealerChunk(content=f"Failed to parse plan: {e}\n")
                if attempt_number < self.max_attempts:
                    current_prompt = f"Your previous response was invalid JSON. Error: {e}. Return ONLY valid JSON."
                    continue
                break

            # specific handling for python to keep 'generated code' display simple if it's just one file
            # otherwise show generic summary
            if len(files) == 1:
                yield HealerChunk(content=f"Generated code:\n{files[0]['content']}\n")
            else:
                yield HealerChunk(
                    content=f"Generated {len(files)} files. Command: {command}\n"
                )

            result = self.executor.execute_project(files, command)
            total_execution_time += result.execution_time_seconds

            if result.success:
                yield HealerChunk(content="\n✓ All tests passed!\n")
                yield HealerChunk(
                    metrics=HealerMetrics(
                        total_attempts=attempt_number,
                        successful=True,
                        total_execution_time_seconds=total_execution_time,
                        final_exit_code=result.exit_code,
                    )
                )
                return

            yield HealerChunk(content="\n✗ Tests failed\n")
            yield HealerChunk(content=f"Error output:\n{result.stderr}\n")

            if attempt_number < self.max_attempts:
                current_prompt = self._build_fix_prompt(
                    json.dumps(plan, indent=2), result.stderr
                )

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

    def _extract_json_plan(self, text: str) -> dict[str, Any]:
        """Extract JSON object from markdown code blocks or plain text."""
        # Try to find JSON block
        pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
        match = re.search(pattern, text, re.DOTALL)

        if match:
            json_str = match.group(1)
        else:
            # Fallback: try to find just the brace structure
            match = re.search(r"\{.*?\}", text, re.DOTALL)
            if match:
                json_str = match.group(0)
            else:
                raise ValueError("No JSON found in response")

        return cast(dict[str, Any], json.loads(json_str))

    def _build_system_prompt(self, available_tools: list[str]) -> str:
        tools_str = ", ".join(available_tools)
        return f"""You are a polyglot code generator.
The following tools are available in the environment: [{tools_str}].

Your task is to generate code and tests for the user's request.
You must return a JSON object with the following structure:

{{
  "files": [
    {{
      "path": "filename.ext",
      "content": "source code..."
    }}
  ],
  "command": "shell command to run tests"
}}

Guidelines:
1. Choose the most appropriate language from the available tools.
2. If Python is used, use 'pytest'.
3. If Rust is used, create a standard project structure (Cargo.toml, src/lib.rs) and use 'cargo test'.
4. If JavaScript is used, use 'node --test' or similar if available.
5. Ensure all code is self-contained.
6. Do NOT include markdown formatting outside the JSON. Return ONLY the JSON object.
"""

    def _build_fix_prompt(self, previous_plan_json: str, error_message: str) -> str:
        return f"""The previous plan failed with this error:

{error_message}

The plan that failed:
```json
{previous_plan_json}
```

Fix the code or command to make all tests pass. Return the corrected JSON object."""
