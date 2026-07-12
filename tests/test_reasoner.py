from unittest.mock import patch

from app.agent.reasoner import decide


def test_decide_parses_valid_json() -> None:
    canned = '{"action": "ingest_and_surface", "reasoning": "test", "related_memory_ids": ["abc"], "conflict_summary": null, "suggested_reminder": null}'
    with patch("app.agent.reasoner.get_llm_provider") as mock_llm:
        mock_llm.return_value.complete.return_value = canned

        result = decide("some clipboard text", [])

    assert result.action == "ingest_and_surface"
    assert result.related_memory_ids == ["abc"]


def test_decide_falls_back_on_malformed_json() -> None:
    with patch("app.agent.reasoner.get_llm_provider") as mock_llm:
        mock_llm.return_value.complete.return_value = "not valid json at all"

        result = decide("some clipboard text", [])

    assert result.action == "ingest_silent"
    assert "Fallback" in result.reasoning
