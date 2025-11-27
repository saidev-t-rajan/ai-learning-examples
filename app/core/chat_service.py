import time
from collections.abc import Generator
from typing import cast

from openai import OpenAI
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletionMessageParam
from app.core.config import Settings
from app.db.chat_repository import ChatRepository
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

    def _get_rag_context(self, message: str) -> tuple[str, float | None, bool]:
        """Retrieve RAG context and return formatted string, avg distance, success flag."""
        if not self.rag_service:
            return "", None, False

        result = self.rag_service.retrieve_context(message)
        return result.formatted_context, result.avg_distance, result.is_success

    def _stream_completion(
        self, messages: list[ChatCompletionMessageParam]
    ) -> Generator[tuple[str, CompletionUsage | None, float], None, None]:
        """Stream LLM response chunks. Yields (content_chunk, usage, ttft)."""
        start_time = time.time()
        ttft = 0.0
        usage_data = None

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
                yield content_chunk, None, ttft

            if chunk.usage:
                usage_data = chunk.usage

        # Final yield with usage data
        if usage_data:
            yield "", usage_data, ttft

    def _build_metrics(
        self,
        ttft: float,
        total_latency: float,
        usage: CompletionUsage | None,
        avg_distance: float | None,
        rag_success: bool,
        response_status: str,
    ) -> ChatMetrics:
        """Build ChatMetrics object from response data."""
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        cost = calculate_cost(self.settings.MODEL_NAME, input_tokens, output_tokens)

        return ChatMetrics(
            ttft=ttft,
            total_latency=total_latency,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            avg_retrieval_distance=avg_distance,
            rag_success=rag_success,
            response_status=response_status,
        )

    def get_response(
        self, message: str, system_message: str | None = None
    ) -> Generator[ChatChunk, None, None]:
        """Stream LLM response chunks, then yield final ChatMetrics."""
        self.repo.add_message("user", message)

        messages, rag_stats = self._prepare_chat_context(message, system_message)

        # State tracking
        state = {
            "full_content": [],
            "ttft": 0.0,
            "usage": None,
            "status": "success",
        }
        start_time = time.time()

        try:
            yield from self._process_stream(messages, state)
        except Exception as e:
            state["status"] = f"error:{str(e)}"

        metrics = self._build_metrics(
            state["ttft"],  # type: ignore
            time.time() - start_time,
            state["usage"],  # type: ignore
            rag_stats["avg_distance"],
            rag_stats["success"],
            str(state["status"]),
        )

        self._save_response("".join(state["full_content"]), metrics)  # type: ignore
        yield ChatChunk(metrics=metrics)

    def _prepare_chat_context(
        self, message: str, system_message: str | None
    ) -> tuple[list[ChatCompletionMessageParam], dict]:
        rag_context, avg_distance, rag_success = self._get_rag_context(message)
        messages = self._prepare_messages(rag_context, system_message)
        return messages, {"avg_distance": avg_distance, "success": rag_success}

    def _process_stream(
        self, messages: list[ChatCompletionMessageParam], state: dict
    ) -> Generator[ChatChunk, None, None]:
        for content, usage, ttft in self._stream_completion(messages):
            if content:
                state["full_content"].append(content)  # type: ignore
                yield ChatChunk(content=content)
            if usage:
                state["usage"] = usage
            if ttft and not state["ttft"]:
                state["ttft"] = ttft

    def _save_response(self, content: str, metrics: ChatMetrics) -> None:
        metrics.message_id = self.repo.add_message(
            role="assistant", content=content, metrics=metrics
        )

    def _prepare_messages(
        self,
        rag_context: str,
        system_message: str | None = None,
    ) -> list[ChatCompletionMessageParam]:
        history = self.repo.get_recent_messages(limit=MAX_HISTORY_MESSAGES)
        system_prompt = system_message or DEFAULT_SYSTEM_PROMPT

        if rag_context:
            system_prompt += rag_context

        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt}
        ]
        for msg in history:
            role = msg.role
            content = msg.content
            messages.append(
                cast(ChatCompletionMessageParam, {"role": role, "content": content})
            )

        return messages
