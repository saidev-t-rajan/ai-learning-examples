import subprocess
import tempfile
from pathlib import Path
from time import time

from app.agents.models import ExecutionResult


class CodeExecutor:
    def __init__(self, work_dir: Path | str | None = None, timeout_seconds: int = 30):
        self.work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp())
        if not self.work_dir.exists():
            self.work_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = timeout_seconds

    def execute_python(self, code: str) -> ExecutionResult:
        test_file_path = self.work_dir / "test_solution.py"
        test_file_path.write_text(code)

        start_time = time()

        try:
            process_result = subprocess.run(
                ["python", "-m", "pytest", str(test_file_path), "-v"],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                cwd=str(self.work_dir),
            )

            execution_time = time() - start_time

            return ExecutionResult(
                success=process_result.returncode == 0,
                stdout=process_result.stdout,
                stderr=process_result.stderr,
                exit_code=process_result.returncode,
                execution_time_seconds=execution_time,
            )

        except subprocess.TimeoutExpired:
            execution_time = time() - start_time
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"Execution timed out after {self.timeout_seconds} seconds",
                exit_code=-1,
                execution_time_seconds=execution_time,
            )

    def execute_rust(self, code: str) -> ExecutionResult:
        raise NotImplementedError("Rust execution not yet implemented")
