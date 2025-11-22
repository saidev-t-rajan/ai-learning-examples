def start_chat():
    print("--- Datacom AI Assessment ---")
    print("Type '/exit' to quit.")

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input == "/exit":
                print("Goodbye!")
                break

            # Placeholder for AI response
            print(f"AI: Echoing '{user_input}'")

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    start_chat()
