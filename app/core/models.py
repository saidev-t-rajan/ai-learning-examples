from enum import IntEnum
from pydantic import BaseModel
from dataclasses import dataclass


class Feedback(IntEnum):
    DOWN = -1
    NEUTRAL = 0
    UP = 1


@dataclass(frozen=True)
class RetrievalResult:
    formatted_context: str
    avg_distance: float | None
    is_success: bool


class ChatMetrics(BaseModel):
    ttft: float = 0.0
    total_latency: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    message_id: int | None = None
    avg_retrieval_distance: float | None = None
    rag_success: bool = False
    response_status: str = "success"


class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: str | None = None


class ChatLogEntry(BaseModel):
    timestamp: str
    metrics: ChatMetrics
    feedback: int | None = None


class ChatChunk(BaseModel):
    content: str | None = None
    metrics: ChatMetrics | None = None
