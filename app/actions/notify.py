from typing import Any

from app.actions.base import Action


class SendNotificationAction(Action):
    name = "send_notification"
    description = "Send an immediate desktop notification to the user for something time-sensitive."
    parameters_schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Short notification title."},
            "message": {"type": "string", "description": "Notification body text."},
        },
        "required": ["title", "message"],
    }

    def execute(self, title: str, message: str) -> dict[str, Any]:
        try:
            from plyer import notification

            notification.notify(title=title, message=message, timeout=10)
            return {"status": "sent", "channel": "desktop"}
        except Exception as exc:
            print(f"[NOTIFY] {title}: {message} (desktop notification unavailable: {exc})")
            return {"status": "logged_fallback", "channel": "log"}
