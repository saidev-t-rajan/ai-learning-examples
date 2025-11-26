import re
from collections.abc import Generator
from pathlib import Path
from typing import cast

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
        system_prompt = self._build_code_generation_prompt()
        current_prompt = task_description

        total_execution_time = 0.0

        for attempt_number in range(1, self.max_attempts + 1):
            yield HealerChunk(
                content=f"\n=== Attempt {attempt_number}/{self.max_attempts} ===\n"
            )

            generated_code = self._generate_code(system_prompt, current_prompt)

            if not generated_code:
                yield HealerChunk(content="Failed to generate code\n")
                break

            yield HealerChunk(content=f"Generated code:\n{generated_code}\n")

            result = self.executor.execute_python(generated_code)
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
                current_prompt = self._build_fix_prompt(generated_code, result.stderr)

        yield HealerChunk(
            metrics=HealerMetrics(
                total_attempts=self.max_attempts,
                successful=False,
                total_execution_time_seconds=total_execution_time,
                final_exit_code=-1,
            )
        )

    def _generate_code(self, system_prompt: str, user_prompt: str) -> str:
        full_response = ""

        for chunk in self.chat_service.get_response(
            message=user_prompt, system_message=system_prompt
        ):
            if chunk.content:
                full_response += chunk.content

        return self._extract_code_from_markdown(full_response)

    def _build_code_generation_prompt(self) -> str:
        return """You are a code generator. Generate ONLY executable Python code with tests.

Requirements:
- Include necessary imports
- Write at least one test function starting with 'test_'
- Use pytest assertions
- Do NOT include explanations
- Wrap code in ```python blocks

Example:
```python
def add(a, b):
    return a + b

def test_add():
    assert add(2, 3) == 5
```"""

    def _build_fix_prompt(self, broken_code: str, error_message: str) -> str:
        return f"""The previous code failed with this error:

{error_message}

The code that failed:
```python
{broken_code}
```

Fix the code to make all tests pass. Return only the corrected code."""

    def _extract_code_from_markdown(self, text: str) -> str:
        pattern = r"```(?:python)?\s*\n(.*?)\n```"
        matches = re.findall(pattern, text, re.DOTALL)

        if matches:
            return cast(str, matches[0]).strip()

        return text.strip()
