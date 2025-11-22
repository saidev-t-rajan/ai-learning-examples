class ChatService:
    def get_response(self, message: str) -> str:
        """
        Get a response for the user's message.

        Currently returns a static string.
        Will be connected to OpenAI in Phase 3.2.
        """
        return f"I received your message: {message}"
