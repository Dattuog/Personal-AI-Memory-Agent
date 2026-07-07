from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.daemon import read_surfaced
from app.ingestion.ingest import ingest_entry
from app.models import IngestRequest, IngestResponse, QueryRequest, QueryResponse, QueryResult
from app.retrieval.retrieve import retrieve_and_rerank
from app.synthesis import get_llm_provider

app = FastAPI(title="PAMA")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    return IngestResponse(ids=ingest_entry(request.text, request.source, request.tags))


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    chunks = retrieve_and_rerank(request.query, request.top_k)
    answer = get_llm_provider().synthesize(request.query, chunks)
    return QueryResponse(answer=answer, sources=[QueryResult(**chunk) for chunk in chunks])


@app.get("/surfaced")
def surfaced() -> list[dict]:
    return read_surfaced()
