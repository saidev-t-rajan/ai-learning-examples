from app.core.chat_service import ChatService
from app.db.memory import ChatRepository


def start_chat():
    print("--- Datacom AI Assessment ---")
    print("Type '/exit' to quit.")

    # Initialize our Service
    repo = ChatRepository()
    chat_service = ChatService(repo=repo)

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input == "/exit":
                print("Goodbye!")
                break

            if not user_input:
                continue

            # Get response from the service (The Logic)
            response = chat_service.get_response(user_input)

            # Print response to the screen (The View)
            print(f"AI: {response}")

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    start_chat()
