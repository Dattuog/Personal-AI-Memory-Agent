from typing import Any

from pydantic import BaseModel, Field


class MemoryEntry(BaseModel):
    id: str
    text: str
    source: str
    timestamp: float
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    text: str
    source: str = "manual"
    tags: list[str] = Field(default_factory=list)


class IngestResponse(BaseModel):
    ids: list[str]


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class QueryResult(BaseModel):
    text: str
    score: float
    cosine_similarity: float
    decay_score: float
    metadata: dict[str, Any]


class QueryResponse(BaseModel):
    answer: str
    action_taken: str | None = None
    action_result: dict[str, Any] | None = None
    sources: list[QueryResult]
    plan: list[str] | None = None
    missing_info: list[str] | None = None
