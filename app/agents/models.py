from pydantic import BaseModel, Field


class TripConstraints(BaseModel):
    destination: str
    duration_days: int
    max_budget_nzd: float
    origin: str = "Wellington"


class Activity(BaseModel):
    name: str
    cost_nzd: float
    description: str


class Accommodation(BaseModel):
    name: str
    nights: int
    cost_per_night_nzd: float
    total_cost_nzd: float


class TripItinerary(BaseModel):
    destination: str
    origin: str
    duration_days: int
    total_cost_nzd: float
    flights: list[dict] = Field(default_factory=list)
    accommodation: list[Accommodation] = Field(default_factory=list)
    activities: list[Activity] = Field(default_factory=list)
    weather_summary: str = ""

    def is_within_budget(self, max_budget: float) -> bool:
        return self.total_cost_nzd <= max_budget


class AgentStep(BaseModel):
    step_type: str
    content: str


class ExecutionResult(BaseModel):
    success: bool
    stdout: str
    stderr: str
    exit_code: int
    execution_time_seconds: float


class HealerMetrics(BaseModel):
    total_attempts: int
    successful: bool
    total_execution_time_seconds: float
    final_exit_code: int


class HealerChunk(BaseModel):
    content: str | None = None
    metrics: HealerMetrics | None = None
