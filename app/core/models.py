from pydantic import BaseModel


class ChatMetrics(BaseModel):
    ttft: float = 0.0
    total_latency: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
