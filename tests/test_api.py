from app.main import ingest, query
from app.models import IngestRequest, QueryRequest


def _source(text: str = "Project notes live in the blue folder.") -> dict:
    return {
        "text": text,
        "score": 0.99,
        "cosine_similarity": 0.98,
        "decay_score": 1.0,
        "metadata": {"source": "manual", "timestamp": 1.0},
    }


def test_health_endpoint() -> None:
    from app.main import health

    assert health() == {"status": "ok"}


def test_ingest_and_query_end_to_end_with_monkeypatch(monkeypatch) -> None:
    stored = []

    def fake_ingest_entry(text: str, source: str, tags: list[str]) -> list[str]:
        stored.append({"text": text, "source": source, "tags": tags})
        return ["memory-1"]

    class FakeLLMProvider:
        def complete_with_tools(self, **kwargs):
            return {"type": "text", "content": f"Your note says: {stored[0]['text']}"}

    monkeypatch.setattr("app.main.ingest_entry", fake_ingest_entry)
    monkeypatch.setattr("app.main.classify_query", lambda query_text: "simple")
    monkeypatch.setattr("app.main.retrieve_and_rerank", lambda query_text, top_k: [_source(stored[0]["text"])])
    monkeypatch.setattr("app.main.get_llm_provider", lambda: FakeLLMProvider())

    ingest_response = ingest(IngestRequest(text="Project notes live in the blue folder."))
    assert ingest_response.model_dump() == {"ids": ["memory-1"]}

    query_response = query(QueryRequest(query="Where are my notes?", top_k=3))
    body = query_response.model_dump()
    assert "blue folder" in body["answer"]
    assert body["sources"][0]["metadata"]["source"] == "manual"

def test_list_memories_endpoint(monkeypatch) -> None:
    from app.main import list_memories

    class FakeStore:
        def list_all(self, limit: int = 1000):
            assert limit == 3
            return [
                {
                    "id": "memory-1",
                    "text": "Project notes live in the blue folder.",
                    "metadata": {"source": "manual"},
                }
            ]

    monkeypatch.setattr("app.main.get_vector_store", lambda: FakeStore())

    assert list_memories(limit=3) == {
        "count": 1,
        "memories": [
            {
                "id": "memory-1",
                "text": "Project notes live in the blue folder.",
                "metadata": {"source": "manual"},
            }
        ],
    }

