from app.main import health, ingest, query
from app.models import IngestRequest, QueryRequest


def test_health_endpoint() -> None:
    assert health() == {"status": "ok"}


def test_ingest_and_query_end_to_end_with_monkeypatch(monkeypatch) -> None:
    stored = []

    def fake_ingest_entry(text: str, source: str, tags: list[str]) -> list[str]:
        stored.append({"text": text, "source": source, "tags": tags})
        return ["memory-1"]

    def fake_retrieve_and_rerank(query: str, top_k: int) -> list[dict]:
        assert query == "Where are my notes?"
        assert top_k == 3
        return [
            {
                "text": stored[0]["text"],
                "score": 0.99,
                "cosine_similarity": 0.98,
                "decay_score": 1.0,
                "metadata": {"source": "manual", "timestamp": 1.0},
            }
        ]

    class FakeLLMProvider:
        def synthesize(self, query: str, chunks: list[dict]) -> str:
            return f"Your note says: {chunks[0]['text']}"

    monkeypatch.setattr("app.main.ingest_entry", fake_ingest_entry)
    monkeypatch.setattr("app.main.retrieve_and_rerank", fake_retrieve_and_rerank)
    monkeypatch.setattr("app.main.get_llm_provider", lambda: FakeLLMProvider())

    ingest_response = ingest(IngestRequest(text="Project notes live in the blue folder."))
    assert ingest_response.model_dump() == {"ids": ["memory-1"]}

    query_response = query(QueryRequest(query="Where are my notes?", top_k=3))
    body = query_response.model_dump()
    assert "blue folder" in body["answer"]
    assert body["sources"][0]["metadata"]["source"] == "manual"
