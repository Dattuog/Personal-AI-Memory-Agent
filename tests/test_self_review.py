import json
from unittest.mock import MagicMock, patch

from app.agent.self_review import run_self_review


def test_self_review_flags_and_creates_reminder(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    canned_response = json.dumps(
        [
            {
                "memory_id": "mem-1",
                "needs_attention": True,
                "reminder_text": "Renew passport before September",
                "reasoning": "Deadline approaching",
            }
        ]
    )
    fake_memory = {"id": "mem-1", "text": "Renew passport before September", "metadata": {"timestamp": 0}}

    with patch("app.agent.self_review.get_vector_store") as mock_store, patch(
        "app.agent.self_review.get_llm_provider"
    ) as mock_llm, patch("app.agent.self_review.get_action") as mock_action:
        mock_store.return_value.list_all.return_value = [fake_memory]
        mock_llm.return_value.complete.return_value = canned_response
        mock_action.return_value.execute = MagicMock()

        results = run_self_review()

    assert results == [{"memory_id": "mem-1", "reminder_text": "Renew passport before September"}]
    mock_action.return_value.execute.assert_called_once_with(text="Renew passport before September")
    mock_store.return_value.update_metadata.assert_called_once()
