"""Request and response models. Structured output keeps the API contract
stable regardless of how the model phrases things."""
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500,
                          examples=["Which invoices are overdue?"])


class AskResponse(BaseModel):
    question: str
    answer: str
    tools_available: list[str]


class HealthResponse(BaseModel):
    status: str
    database: str
    policies_loaded: int
    model_configured: bool
