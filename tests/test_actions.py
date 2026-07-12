import json
from pathlib import Path

from app.actions.reminder import CreateReminderAction


def test_create_reminder_persists(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    action = CreateReminderAction()

    result = action.execute(text="Renew passport", due_date="2026-09-01")

    assert result["status"] == "created"
    saved = json.loads(Path("reminders.json").read_text(encoding="utf-8"))
    assert len(saved) == 1
    assert saved[0]["text"] == "Renew passport"
    assert saved[0]["due_date"] == "2026-09-01"
    assert saved[0]["completed"] is False
