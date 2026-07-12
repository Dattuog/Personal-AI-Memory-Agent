from typing import Literal

from pydantic import BaseModel, Field


class AgentDecision(BaseModel):
    action: Literal[
        "ingest_silent",
        "ingest_and_surface",
        "ingest_and_flag_conflict",
        "skip",
        "self_review_flag",
    ]
    reasoning: str
    related_memory_ids: list[str] = Field(default_factory=list)
    conflict_summary: str | None = None
    suggested_reminder: str | None = None
