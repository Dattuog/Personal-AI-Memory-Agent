from app.actions.base import Action
from app.actions.notify import SendNotificationAction
from app.actions.reminder import CreateReminderAction

_ACTIONS: dict[str, Action] = {
    "create_reminder": CreateReminderAction(),
    "send_notification": SendNotificationAction(),
}


def get_action(name: str) -> Action:
    if name not in _ACTIONS:
        raise ValueError(f"Unknown action: {name}")
    return _ACTIONS[name]


def all_tool_specs() -> list[dict]:
    return [action.to_tool_spec() for action in _ACTIONS.values()]
