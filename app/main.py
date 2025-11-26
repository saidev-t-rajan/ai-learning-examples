import time
from typing import Callable
from dataclasses import dataclass
from app.core.config import Settings
from app.core.chat_service import ChatService
from app.db.memory import ChatRepository
from app.rag.service import RAGService


# Command constants
EXIT = "/exit"
INGEST = "/ingest"
INGEST_ALL = "/ingest_all"

SEPARATOR_LINE = "-" * 30


def ingest_directory_with_report(service: RAGService, directory_path: str) -> None:
    """
    Ingests all documents from a directory using the provided RAGService
    and prints progress report to stdout.
    """
    print(f"Ingesting documents from {directory_path}...")
    start_time = time.time()

    try:
        results_generator = service.ingest_directory(directory_path)
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


@dataclass
class Command:
    prefix: str
    handler: Callable[[str, RAGService | None, Settings], None]
    requires_rag: bool = False
    exact_match: bool = False


def _handle_exit(user_input: str, rag: RAGService | None, settings: Settings) -> None:
    print("Goodbye!")


def _handle_ingest_all(
    user_input: str, rag: RAGService | None, settings: Settings
) -> None:
    assert rag is not None
    target_dir = settings.CORPUS_DIR
    if " --large" in user_input:
        target_dir = settings.CORPUS_LARGE_DIR
    ingest_directory_with_report(rag, target_dir)


def _handle_ingest(user_input: str, rag: RAGService | None, settings: Settings) -> None:
    assert rag is not None
    path = user_input.split(" ", 1)[1].strip()
    count = rag.ingest(path)
    print(f"Ingested {count} chunks from {path}")


COMMANDS = [
    Command(EXIT, _handle_exit, exact_match=True),
    Command(INGEST_ALL, _handle_ingest_all, requires_rag=True),
    Command(INGEST + " ", _handle_ingest, requires_rag=True),
]


def _handle_command(
    user_input: str, rag_service: RAGService | None, settings: Settings
) -> bool:
    for cmd in COMMANDS:
        is_match = False
        if cmd.exact_match:
            if user_input == cmd.prefix:
                is_match = True
        else:
            if user_input.startswith(cmd.prefix):
                is_match = True

        if is_match:
            if cmd.requires_rag and not rag_service:
                print("RAG service not available.")
                return True

            cmd.handler(user_input, rag_service, settings)
            return True

    return False


def _process_chat(chat_service: ChatService, user_input: str) -> None:
    print("AI: ", end="", flush=True)

    response_generator = chat_service.get_response(user_input)

    for chunk in response_generator:
        if chunk.content:
            print(chunk.content, end="", flush=True)
        if chunk.metrics:
            print(chunk.metrics.format_stats())

    print()  # Newline at end


def start_chat() -> None:
    settings = Settings()
    print("--- Datacom AI Assessment ---")
    print(f"Type '{EXIT}' to quit.")
    print(f"Type '{INGEST} <path>' to load a document.")
    print(f"Type '{INGEST_ALL} [--large]' to load all corpus documents.")

    # Initialize our Service
    repo = ChatRepository()
    rag_service = RAGService(settings=settings)
    chat_service = ChatService(repo=repo, rag_service=rag_service, settings=settings)

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if _handle_command(user_input, rag_service, settings):
                if user_input == EXIT:
                    break
                continue

            _process_chat(chat_service, user_input)

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    start_chat()
