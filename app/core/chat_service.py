import time
from collections.abc import Generator
from typing import cast

from openai import OpenAI
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletionMessageParam
from app.core.config import Settings
from app.db.memory import ChatRepository
from app.core.models import ChatMetrics, ChatChunk
from app.core.utils import calculate_cost
from app.rag.service import RAGService


MAX_HISTORY_MESSAGES = 10
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


class ChatService:
    def __init__(
        self,
        repo: ChatRepository,
        settings: Settings | None = None,
        rag_service: RAGService | None = None,
    ):
        self.settings = settings or Settings()
        self.repo = repo
        self.rag_service = rag_service
        self.client = OpenAI(
            api_key=self.settings.OPENAI_API_KEY,
            base_url=self.settings.OPENAI_BASE_URL,
        )

    def get_response(self, message: str) -> Generator[ChatChunk, None, None]:
        """
        Get a streaming response for the user's message.
        Yields string chunks and finally a ChatMetrics object.
        """
        self.repo.add_message("user", message)

        # 1. Retrieve Context & Scores
        rag_context = ""
        avg_distance: float | None = None
        rag_success = False

        if self.rag_service:
            retrieval_result = self.rag_service.retrieve_context(message)
            rag_context = retrieval_result.formatted_context
            avg_distance = retrieval_result.avg_distance
            rag_success = retrieval_result.is_success

        # 2. Prepare Messages
        messages = self._prepare_messages(rag_context)

        start_time = time.time()
        ttft = 0.0
        full_content = []
        usage_data: CompletionUsage | None = None
        response_status = "success"

        try:
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
                    yield ChatChunk(content=content_chunk)

                if chunk.usage:
                    usage_data = chunk.usage

        except Exception as e:
            response_status = f"error:{str(e)}"

        end_time = time.time()

        final_response = "".join(full_content)

        # Calculate metrics first to pass to DB
        input_tokens = usage_data.prompt_tokens if usage_data else 0
        output_tokens = usage_data.completion_tokens if usage_data else 0
        cost = calculate_cost(self.settings.MODEL_NAME, input_tokens, output_tokens)
        total_latency = end_time - start_time

        # Persist assistant response with all metrics
        metrics = ChatMetrics(
            ttft=ttft,
            total_latency=total_latency,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            avg_retrieval_distance=avg_distance,
            rag_success=rag_success,
            response_status=response_status,
        )

        message_id = self.repo.add_message(
            role="assistant",
            content=final_response,
            metrics=metrics,
        )

        metrics.message_id = message_id
        yield ChatChunk(metrics=metrics)

    def _prepare_messages(
        self,
        rag_context: str,
    ) -> list[ChatCompletionMessageParam]:
        history = self.repo.get_recent_messages(limit=MAX_HISTORY_MESSAGES)
        system_prompt = DEFAULT_SYSTEM_PROMPT

        if rag_context:
            system_prompt += rag_context

        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt}
        ]
        for msg in history:
            role = msg["role"]
            content = msg["content"]
            messages.append(
                cast(ChatCompletionMessageParam, {"role": role, "content": content})
            )

        return messages
