import time

from app.core.config import Settings
from app.core.chat_service import ChatService
from app.rag.service import RAGService


# Command constants
EXIT = "/exit"
INGEST = "/ingest"
INGEST_ALL = "/ingest_all"

SEPARATOR_LINE = "-" * 30


def ingest_directory_with_report(rag_service: RAGService, directory_path: str) -> None:
    """
    Ingests all documents from a directory using the RAGService
    and prints progress report to stdout.
    """
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
                # Errors are already printed by the service, so we just mark it here
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
    ):
        self.chat_service = chat_service
        self.rag_service = rag_service
        self.settings = settings

    def run(self) -> None:
        """Start the interactive chat loop."""
        print("--- Datacom AI Assessment ---")
        print(f"Type '{EXIT}' to quit.")
        print(f"Type '{INGEST} <path>' to load a document.")
        print(f"Type '{INGEST_ALL} [--large]' to load all corpus documents.")

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

    def _process_chat(self, user_input: str) -> None:
        print("AI: ", end="", flush=True)

        response_generator = self.chat_service.get_response(user_input)

        for chunk in response_generator:
            if chunk.content:
                print(chunk.content, end="", flush=True)
            if chunk.metrics:
                print(chunk.metrics.format_stats())

        print()  # Newline at end
