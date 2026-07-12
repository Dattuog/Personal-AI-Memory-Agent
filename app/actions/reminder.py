import json
import time
import uuid
from pathlib import Path
from typing import Any

from app.actions.base import Action

REMINDERS_PATH = Path("reminders.json")


class CreateReminderAction(Action):
    name = "create_reminder"
    description = (
        "Create a reminder for the user about something they need to do or remember, "
        "optionally tied to a due date."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "What to remind the user about."},
            "due_date": {
                "type": "string",
                "description": "ISO 8601 date (YYYY-MM-DD) if known, otherwise omit.",
            },
        },
        "required": ["text"],
    }

    def execute(self, text: str, due_date: str | None = None) -> dict[str, Any]:
        reminders = []
        if REMINDERS_PATH.exists():
            reminders = json.loads(REMINDERS_PATH.read_text(encoding="utf-8"))

        reminder = {
            "id": str(uuid.uuid4()),
            "text": text,
            "due_date": due_date,
            "created_at": time.time(),
            "completed": False,
        }
        reminders.append(reminder)
        REMINDERS_PATH.write_text(json.dumps(reminders, indent=2), encoding="utf-8")
        return {"status": "created", "reminder": reminder}
