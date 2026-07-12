import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.actions.registry import all_tool_specs, get_action
from app.agent.planner import classify_query, run_multi_step_task
from app.agent.self_review import run_self_review
from app.daemon import read_surfaced
from app.ingestion.ingest import ingest_entry
from app.models import IngestRequest, IngestResponse, QueryRequest, QueryResponse, QueryResult
from app.retrieval.retrieve import retrieve_and_rerank
from app.storage import get_vector_store
from app.synthesis import get_llm_provider

SIMPLE_QUERY_SYSTEM_PROMPT = """You are a personal memory assistant. You can either answer the user's question using the provided memories, or call a tool if the user is asking you to remember or remind them of something actionable.

Only call a tool if the request clearly asks for an action. Otherwise, answer directly using the memories provided."""

app = FastAPI(title="PAMA")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _to_query_results(chunks: list[dict[str, Any]]) -> list[QueryResult]:
    return [QueryResult(**chunk) for chunk in chunks]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    return IngestResponse(ids=ingest_entry(request.text, request.source, request.tags))


def handle_simple_query(query_text: str, top_k: int) -> QueryResponse:
    chunks = retrieve_and_rerank(query_text, top_k)
    llm = get_llm_provider()
    context_text = "\n\n".join(f"[{chunk['metadata'].get('source', 'unknown')}] {chunk['text']}" for chunk in chunks)
    user_message = f"Memories:\n{context_text}\n\nUser request: {query_text}"

    result = llm.complete_with_tools(
        system_prompt=SIMPLE_QUERY_SYSTEM_PROMPT,
        user_message=user_message,
        tools=all_tool_specs(),
    )

    if result["type"] == "tool_call":
        action = get_action(result["name"])
        action_result = action.execute(**result["arguments"])
        return QueryResponse(
            answer=f"Done - {result['name'].replace('_', ' ')} action taken.",
            action_taken=result["name"],
            action_result=action_result,
            sources=_to_query_results(chunks),
        )

    content = result.get("content") or llm.synthesize(query_text, chunks)
    return QueryResponse(answer=content, sources=_to_query_results(chunks))


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    complexity = classify_query(request.query)

    if complexity == "complex":
        result = run_multi_step_task(request.query)
        return QueryResponse(
            answer=result["answer"],
            sources=_to_query_results(result["sources"]),
            plan=result["plan"],
            missing_info=result["missing_info"],
        )

    return handle_simple_query(request.query, request.top_k)


@app.get("/surfaced")
def surfaced() -> list[dict]:
    return read_surfaced()


@app.get("/memories")
def list_memories(limit: int = 200) -> dict[str, Any]:
    store = get_vector_store()
    memories = store.list_all(limit=limit)
    return {"count": len(memories), "memories": memories}


@app.get("/reminders")
def list_reminders() -> dict[str, list[dict[str, Any]]]:
    path = Path("reminders.json")
    if not path.exists():
        return {"reminders": []}
    return {"reminders": json.loads(path.read_text(encoding="utf-8"))}


@app.post("/self-review/run")
def trigger_self_review() -> dict[str, Any]:
    flagged = run_self_review()
    return {"flagged_count": len(flagged), "flagged": flagged}

