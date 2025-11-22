from openai import OpenAI
from app.core.config import Settings
from app.db.memory import ChatRepository


class ChatService:
    def __init__(self, repo: ChatRepository, settings: Settings = None):
        self.settings = settings or Settings()
        self.repo = repo
        self.client = OpenAI(
            api_key=self.settings.OPENAI_API_KEY,
            base_url=self.settings.OPENAI_BASE_URL,
        )

    def get_response(self, message: str) -> str:
        """
        Get a response for the user's message.
        """
        # 1. Persist User Message
        self.repo.add_message("user", message)

        # 2. Fetch Context (including the message we just saved)
        history = self.repo.get_recent_messages(limit=10)

        # 3. Prepare Messages for LLM
        messages = [{"role": "system", "content": "You are a helpful assistant."}]

        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})

        # 4. Call LLM
        response = self.client.chat.completions.create(
            model=self.settings.MODEL_NAME,
            messages=messages,
        )
        content = response.choices[0].message.content

        # 5. Persist AI Response
        self.repo.add_message("assistant", content)

        return content
