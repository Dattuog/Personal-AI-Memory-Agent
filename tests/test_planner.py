import json
from unittest.mock import patch

from app.agent.planner import build_plan, classify_query, execute_plan


def test_classify_query_complex() -> None:
    with patch("app.agent.planner.get_llm_provider") as mock_llm:
        mock_llm.return_value.complete.return_value = "complex"

        assert classify_query("help me plan my September trip") == "complex"


def test_classify_query_defaults_to_simple() -> None:
    with patch("app.agent.planner.get_llm_provider") as mock_llm:
        mock_llm.return_value.complete.return_value = "simple"

        assert classify_query("when is my dentist appointment") == "simple"


def test_build_plan_parses_json_list() -> None:
    canned = json.dumps(["What flights are booked?", "What hotel is booked?", "Any visa requirements noted?"])
    with patch("app.agent.planner.get_llm_provider") as mock_llm:
        mock_llm.return_value.complete.return_value = canned

        steps = build_plan("help me plan my September trip")

    assert steps == ["What flights are booked?", "What hotel is booked?", "Any visa requirements noted?"]


def test_build_plan_falls_back_on_bad_json() -> None:
    with patch("app.agent.planner.get_llm_provider") as mock_llm:
        mock_llm.return_value.complete.return_value = "not json"

        steps = build_plan("help me plan my September trip")

    assert steps == ["help me plan my September trip"]


def test_execute_plan_flags_missing_data() -> None:
    with patch("app.agent.planner.retrieve_and_rerank") as mock_retrieve:
        mock_retrieve.side_effect = [[{"text": "Booked flights"}], []]

        results = execute_plan(["flights?", "hotel?"])

    assert results[0]["has_data"] is True
    assert results[1]["has_data"] is False
