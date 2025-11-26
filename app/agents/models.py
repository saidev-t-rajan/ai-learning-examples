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
