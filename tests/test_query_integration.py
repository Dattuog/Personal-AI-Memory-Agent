from unittest.mock import MagicMock

from app.models import QueryRequest


def _source(text: str = "Remember to book a hotel.") -> dict:
    return {
        "text": text,
        "score": 0.91,
        "cosine_similarity": 0.9,
        "decay_score": 0.95,
        "metadata": {"source": "manual", "timestamp": 1.0},
    }


def test_query_routes_complex(monkeypatch) -> None:
    from app.main import query

    monkeypatch.setattr("app.main.classify_query", lambda query_text: "complex")
    monkeypatch.setattr(
        "app.main.run_multi_step_task",
        lambda query_text: {
            "answer": "Flights found. Still missing hotel.",
            "sources": [_source("Flight booked for September")],
            "plan": ["What flights are booked?", "What hotel is booked?"],
            "missing_info": ["What hotel is booked?"],
        },
    )

    response = query(QueryRequest(query="help me plan my September trip"))

    assert response.plan == ["What flights are booked?", "What hotel is booked?"]
    assert response.missing_info == ["What hotel is booked?"]


def test_query_simple_tool_call_executes_action(tmp_path, monkeypatch) -> None:
    from app.main import query

    class FakeLLMProvider:
        def complete_with_tools(self, **kwargs):
            return {"type": "tool_call", "name": "create_reminder", "arguments": {"text": "Book a hotel"}}

    fake_action = MagicMock()
    fake_action.execute.return_value = {"status": "created", "reminder": {"text": "Book a hotel"}}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("app.main.classify_query", lambda query_text: "simple")
    monkeypatch.setattr("app.main.retrieve_and_rerank", lambda query_text, top_k: [_source()])
    monkeypatch.setattr("app.main.get_llm_provider", lambda: FakeLLMProvider())
    monkeypatch.setattr("app.main.get_action", lambda name: fake_action)

    response = query(QueryRequest(query="remind me to book a hotel"))

    assert response.action_taken == "create_reminder"
    assert response.action_result["status"] == "created"
    fake_action.execute.assert_called_once_with(text="Book a hotel")
