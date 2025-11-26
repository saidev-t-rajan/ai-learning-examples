from pydantic import BaseModel


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

    def format_stats(self) -> str:
        """Format metrics for CLI display."""
        return (
            f"\n\n[Stats] "
            f"Prompt: {self.input_tokens} | "
            f"Completion: {self.output_tokens} | "
            f"TTFT: {self.ttft:.2f}s | "
            f"Latency: {self.total_latency:.2f}s | "
            f"Cost: ${self.cost:.6f}"
        )


class ChatChunk(BaseModel):
    content: str | None = None
    metrics: ChatMetrics | None = None
