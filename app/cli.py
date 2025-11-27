import time

from app.core.config import Settings
from app.core.chat_service import ChatService
from app.rag.service import RAGService
from app.agents.models import HealerMetrics
from app.agents.planning import PlanningService
from app.agents.healer import HealerService
from app.core.models import ChatMetrics

# Command constants
EXIT = "/exit"
INGEST = "/ingest"
INGEST_ALL = "/ingest_all"
PLAN = "/plan"
HEAL = "/heal"

SEPARATOR_LINE = "-" * 30


def ingest_directory_with_report(rag_service: RAGService, directory_path: str) -> None:
    print(f"Ingesting documents from {directory_path}...")
    start_time = time.time()

    try:
        results_generator = rag_service.ingest_directory(directory_path)
        total_chunks = 0

        for i, (filename, chunks) in enumerate(results_generator, 1):
            if chunks > 0:
                print(f"[{i}] {filename}: {chunks} chunks", flush=True)
                total_chunks += chunks
            else:
                print(f"[{i}] {filename}: Failed or Empty", flush=True)

    except ValueError as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(f"Unexpected Error: {e}")
        return

    end_time = time.time()
    duration = end_time - start_time

    print(SEPARATOR_LINE)
    print("Ingestion Complete.")
    print(f"Total Time:   {duration:.2f}s")
    print(f"Total Chunks: {total_chunks}")
    if duration > 0:
        print(f"Avg Speed:    {total_chunks / duration:.1f} chunks/sec")
    else:
        print("Avg Speed:    N/A chunks/sec")


class CLI:
    def __init__(
        self,
        chat_service: ChatService,
        rag_service: RAGService,
        settings: Settings,
        planning_service: PlanningService | None = None,
        healer_service: HealerService | None = None,
    ):
        self.chat_service = chat_service
        self.rag_service = rag_service
        self.settings = settings
        self.planning_service = planning_service
        self.healer_service = healer_service

    def run(self) -> None:
        print("--- Datacom AI Assessment ---")
        print(f"Type '{EXIT}' to quit.")
        print(f"Type '{INGEST} <path>' to load a document.")
        print(f"Type '{INGEST_ALL} [--large]' to load all corpus documents.")
        print(f"Type '{PLAN} <request>' to plan a trip with AI agent.")
        print(f"Type '{HEAL} <task>' to generate and fix code with AI.")

        while True:
            try:
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                if self._handle_command(user_input):
                    if user_input == EXIT:
                        break
                    continue

                self._process_chat(user_input)

            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break

    def _handle_command(self, user_input: str) -> bool:
        if user_input == EXIT:
            self._handle_exit()
            return True

        if user_input.startswith(INGEST_ALL):
            self._handle_ingest_all(user_input)
            return True

        if user_input.startswith(INGEST + " "):
            self._handle_ingest(user_input)
            return True

        if user_input.startswith(PLAN + " "):
            self._handle_plan(user_input)
            return True

        if user_input.startswith(HEAL):
            task_description = user_input[len(HEAL) :].strip()
            if not task_description:
                print("Usage: /heal <coding task description>")
                return True

            self._handle_heal_command(task_description)
            return True

        return False

    def _handle_exit(self) -> None:
        print("Goodbye!")

    def _handle_ingest_all(self, user_input: str) -> None:
        target_dir = self.settings.CORPUS_DIR
        if " --large" in user_input:
            target_dir = self.settings.CORPUS_LARGE_DIR
        ingest_directory_with_report(self.rag_service, target_dir)

    def _handle_ingest(self, user_input: str) -> None:
        path = user_input.split(" ", 1)[1].strip()
        count = self.rag_service.ingest(path)
        print(f"Ingested {count} chunks from {path}")

    def _handle_plan(self, user_input: str) -> None:
        if not self.planning_service:
            print("Planning service not available")
            return

        request = user_input[len(PLAN) :].strip()
        print("Agent: ", flush=True)

        for step in self.planning_service.plan(request):
            if step.step_type == "thought":
                print(f"[Thinking] {step.content}", flush=True)
            elif step.step_type == "tool_call":
                print(f"[Tool Call] {step.content}", flush=True)
            elif step.step_type == "tool_result":
                print(f"[Tool Result] {step.content[:100]}...", flush=True)
            elif step.step_type == "final_answer":
                print(f"\n[Final Plan]\n{step.content}", flush=True)
            elif step.step_type == "metrics":
                print(f"\n[Stats] {step.content}", flush=True)

        print()

    def _handle_heal_command(self, task_description: str) -> None:
        if not self.healer_service:
            print("Healer service not available")
            return

        for chunk in self.healer_service.heal_code(task_description):
            if chunk.content:
                print(chunk.content, end="", flush=True)
            if chunk.metrics:
                self._print_healer_metrics(chunk.metrics)

    def _print_healer_metrics(self, metrics: HealerMetrics) -> None:
        status = "SUCCESS" if metrics.successful else "FAILED"
        print(f"\n[{status}] Attempts: {metrics.total_attempts}")
        print(f"Total time: {metrics.total_execution_time_seconds:.2f}s")

    def _process_chat(self, user_input: str) -> None:
        print("AI: ", end="", flush=True)

        response_generator = self.chat_service.get_response(user_input)

        for chunk in response_generator:
            if chunk.content:
                print(chunk.content, end="", flush=True)
            if chunk.metrics:
                print(format_chat_metrics(chunk.metrics))

        print()  # Newline at end


def format_chat_metrics(metrics: ChatMetrics) -> str:
    """Format metrics for CLI display."""
    return (
        f"\n\n[Stats] "
        f"Prompt: {metrics.input_tokens} | "
        f"Completion: {metrics.output_tokens} | "
        f"TTFT: {metrics.ttft:.2f}s | "
        f"Latency: {metrics.total_latency:.2f}s | "
        f"Cost: ${metrics.cost:.6f}"
    )
