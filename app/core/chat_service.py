import time
from typing import Generator, Union
from openai import OpenAI
from app.core.config import Settings
from app.db.memory import ChatRepository
from app.core.models import ChatMetrics
from app.core.utils import calculate_cost


class ChatService:
    def __init__(self, repo: ChatRepository, settings: Settings = None):
        self.settings = settings or Settings()
        self.repo = repo
        self.client = OpenAI(
            api_key=self.settings.OPENAI_API_KEY,
            base_url=self.settings.OPENAI_BASE_URL,
        )

    def get_response(
        self, message: str
    ) -> Generator[Union[str, ChatMetrics], None, None]:
        """
        Get a streaming response for the user's message.
        Yields string chunks and finally a ChatMetrics object.
        """
        # 1. Persist User Message
        self.repo.add_message("user", message)

        # 2. Fetch Context
        history = self.repo.get_recent_messages(limit=10)

        # 3. Prepare Messages
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})

        start_time = time.time()
        ttft = 0.0
        full_content = []
        usage_data = None

        # 4. Call LLM with streaming
        stream = self.client.chat.completions.create(
            model=self.settings.MODEL_NAME,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
        )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content_chunk = chunk.choices[0].delta.content
                if not ttft:
                    ttft = time.time() - start_time

                full_content.append(content_chunk)
                yield content_chunk

            if chunk.usage:
                usage_data = chunk.usage

        end_time = time.time()
        total_latency = end_time - start_time

        # Combine content
        final_response = "".join(full_content)

        # 5. Persist AI Response
        self.repo.add_message("assistant", final_response)

        # Calculate Metrics
        input_tokens = usage_data.prompt_tokens if usage_data else 0
        output_tokens = usage_data.completion_tokens if usage_data else 0
        cost = calculate_cost(self.settings.MODEL_NAME, input_tokens, output_tokens)

        metrics = ChatMetrics(
            ttft=ttft,
            total_latency=total_latency,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
        )

        yield metrics
