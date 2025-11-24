import time
from collections.abc import Generator

from openai import OpenAI
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletionMessageParam
from app.core.config import Settings
from app.db.memory import ChatRepository
from app.core.models import ChatMetrics
from app.core.utils import calculate_cost
from app.rag.service import RAGService


MAX_HISTORY_MESSAGES = 10
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
RAG_CONTEXT_TEMPLATE = (
    "\n\nUse the following context to answer the question.\n"
    "Answer using ONLY the following context. Cite sources using the format [1], [2], etc.\n\n"
    "{context}"
)


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

    def get_response(self, message: str) -> Generator[str | ChatMetrics, None, None]:
        """
        Get a streaming response for the user's message.
        Yields string chunks and finally a ChatMetrics object.
        """
        self.repo.add_message("user", message)

        messages = self._prepare_messages(message)

        start_time = time.time()
        ttft = 0.0
        full_content = []
        usage_data: CompletionUsage | None = None

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

        final_response = "".join(full_content)

        self.repo.add_message("assistant", final_response)

        metrics = self._create_metrics(start_time, end_time, ttft, usage_data)
        yield metrics

    def _prepare_messages(self, message: str) -> list[ChatCompletionMessageParam]:
        history = self.repo.get_recent_messages(limit=MAX_HISTORY_MESSAGES)
        system_prompt = DEFAULT_SYSTEM_PROMPT

        if self.rag_service:
            context_results = self.rag_service.retrieve(message)
            if context_results:
                formatted_context = "\n\n".join(
                    f"[{i}] (Source: {meta.get('source', 'Unknown')})\n{text}"
                    for i, (text, meta) in enumerate(context_results, 1)
                )
                system_prompt += RAG_CONTEXT_TEMPLATE.format(context=formatted_context)

        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt}
        ]
        for msg in history:
            messages.append(
                {"role": msg["role"], "content": msg["content"]}  # type: ignore
            )

        return messages

    def _create_metrics(
        self,
        start_time: float,
        end_time: float,
        ttft: float,
        usage_data: CompletionUsage | None,
    ) -> ChatMetrics:
        total_latency = end_time - start_time
        input_tokens = usage_data.prompt_tokens if usage_data else 0
        output_tokens = usage_data.completion_tokens if usage_data else 0
        cost = calculate_cost(self.settings.MODEL_NAME, input_tokens, output_tokens)

        return ChatMetrics(
            ttft=ttft,
            total_latency=total_latency,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
        )
