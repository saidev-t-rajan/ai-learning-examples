import time
from app.core.config import Settings
from app.core.chat_service import ChatService
from app.db.memory import ChatRepository
from app.core.models import ChatMetrics
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


def start_chat() -> None:
    settings = Settings()
    print("--- Datacom AI Assessment ---")
    print(f"Type '{EXIT}' to quit.")
    print(f"Type '{INGEST} <path>' to load a document.")
    print(f"Type '{INGEST_ALL}' to load all corpus documents.")

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


def _handle_command(
    user_input: str, rag_service: RAGService | None, settings: Settings
) -> bool:
    if user_input == EXIT:
        print("Goodbye!")
        return True

    if user_input == INGEST_ALL:
        if rag_service:
            ingest_directory_with_report(rag_service, settings.CORPUS_DIR)
        else:
            print("RAG service not available.")
        return True

    if user_input.startswith(INGEST + " "):
        path = user_input.split(" ", 1)[1].strip()
        if rag_service:
            count = rag_service.ingest(path)
            print(f"Ingested {count} chunks from {path}")
        else:
            print("RAG service not available.")
        return True

    return False


def _format_stats(metrics: ChatMetrics) -> str:
    """Format chat metrics for display."""
    return (
        f"\n\n[Stats] "
        f"Prompt: {metrics.input_tokens} | "
        f"Completion: {metrics.output_tokens} | "
        f"TTFT: {metrics.ttft:.2f}s | "
        f"Latency: {metrics.total_latency:.2f}s | "
        f"Cost: ${metrics.cost:.6f}"
    )


def _process_chat(chat_service: ChatService, user_input: str) -> None:
    print("AI: ", end="", flush=True)

    response_generator = chat_service.get_response(user_input)

    for chunk in response_generator:
        if isinstance(chunk, str):
            print(chunk, end="", flush=True)
        elif isinstance(chunk, ChatMetrics):
            print(_format_stats(chunk))

    print()  # Newline at end


if __name__ == "__main__":
    start_chat()
