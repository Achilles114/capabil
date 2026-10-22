from pydantic import BaseModel, Field
from typing import Literal

class Capability(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    proficiency: int = Field(ge=1, le=5)
    evidence: str = Field(default="", max_length=500)

class AnalyzeRequest(BaseModel):
    subject_id: str = Field(default="demo-subject", min_length=1, max_length=80)
    current_capabilities: list[Capability]
    target: str = Field(min_length=1, max_length=120)

class AssistantRequest(AnalyzeRequest):
    question: str = Field(min_length=5, max_length=1000)

class FeedbackRequest(BaseModel):
    trace_id: str = Field(min_length=1, max_length=100)
    recommendation_id: str = Field(min_length=1, max_length=100)
    decision: Literal["accepted", "rejected", "modified"]
    reason: str = Field(default="", max_length=500)
    reviewer: str = Field(default="demo-reviewer", max_length=100)
