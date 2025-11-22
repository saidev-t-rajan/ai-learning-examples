from openai import OpenAI
from app.core.config import Settings


class ChatService:
    def __init__(self):
        self.settings = Settings()
        self.client = OpenAI(
            api_key=self.settings.OPENAI_API_KEY,
            base_url=self.settings.OPENAI_BASE_URL,
        )

    def get_response(self, message: str) -> str:
        """
        Get a response for the user's message.
        """
        response = self.client.chat.completions.create(
            model=self.settings.MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": message},
            ],
        )
        return response.choices[0].message.content
