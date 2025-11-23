from app.core.chat_service import ChatService
from app.db.memory import ChatRepository
from app.core.models import ChatMetrics
from app.rag.service import RAGService


def start_chat():
    print("--- Datacom AI Assessment ---")
    print("Type '/exit' to quit.")
    print("Type '/ingest <path>' to load a document.")

    # Initialize our Service
    repo = ChatRepository()
    rag_service = RAGService()
    chat_service = ChatService(repo=repo, rag_service=rag_service)

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input == "/exit":
                print("Goodbye!")
                break

            if not user_input:
                continue

            if user_input.startswith("/ingest "):
                path = user_input.split(" ", 1)[1].strip()
                count = rag_service.ingest(path)
                print(f"Ingested {count} chunks from {path}")
                continue

            # Get response from the service (The Logic)
            print("AI: ", end="", flush=True)

            response_generator = chat_service.get_response(user_input)

            for chunk in response_generator:
                if isinstance(chunk, str):
                    print(chunk, end="", flush=True)
                elif isinstance(chunk, ChatMetrics):
                    print(
                        f"\n\n\n[Metrics] Prompt: {chunk.input_tokens} | Completion: {chunk.output_tokens} | TTFT: {chunk.ttft:.2f}s | Total: {chunk.total_latency:.2f}s | Cost: ${chunk.cost:.6f}"
                    )

            print()  # Newline at end

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    start_chat()
