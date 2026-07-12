# PAMA — Upgrade: Action-Taking (Reminders & Notifications)

## Goal

Right now PAMA only ever returns text — either an answer from `/query` or a logged decision from the daemon. This upgrade gives it actual **tools it can invoke**: creating reminders and sending notifications. The LLM decides *when* to use a tool (via function/tool calling), and the app executes the action for real. This is what turns PAMA from "answers questions about your notes" into "does something for you."

Groq's chat completions API is OpenAI-compatible and supports tool calling on supported models (e.g., `llama-3.3-70b-versatile`), so we use native tool calling rather than hand-rolled JSON parsing like the reasoning daemon used.

---

## 1. Define the Action interface

**`app/actions/base.py`:**
```python
from abc import ABC, abstractmethod
from typing import Any, Dict

class Action(ABC):
    name: str
    description: str
    parameters_schema: Dict[str, Any]  # JSON schema for the tool's parameters

    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Run the action, return a result dict to report back to the LLM/user."""
        ...

    def to_tool_spec(self) -> Dict[str, Any]:
        """Convert to the OpenAI/Groq tool-calling function spec format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }
```

## 2. Implement concrete actions

**`app/actions/reminder.py`:**
```python
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict
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
                "description": "ISO 8601 date (YYYY-MM-DD) if a specific due date is known, otherwise omit.",
            },
        },
        "required": ["text"],
    }

    def execute(self, text: str, due_date: str | None = None) -> Dict[str, Any]:
        reminders = []
        if REMINDERS_PATH.exists():
            reminders = json.loads(REMINDERS_PATH.read_text())

        reminder = {
            "id": str(uuid.uuid4()),
            "text": text,
            "due_date": due_date,
            "created_at": time.time(),
            "completed": False,
        }
        reminders.append(reminder)
        REMINDERS_PATH.write_text(json.dumps(reminders, indent=2))
        return {"status": "created", "reminder": reminder}
```

**`app/actions/notify.py`:**
```python
from typing import Any, Dict
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

    def execute(self, title: str, message: str) -> Dict[str, Any]:
        try:
            from plyer import notification
            notification.notify(title=title, message=message, timeout=10)
            return {"status": "sent", "channel": "desktop"}
        except Exception as e:
            # Fallback: log-only, keeps this working headless/in Docker without crashing
            print(f"[NOTIFY] {title}: {message} (desktop notification unavailable: {e})")
            return {"status": "logged_fallback", "channel": "log"}
```

Add `plyer` to `requirements.txt`. On Linux desktops it uses `notify-send` under the hood; in Docker/headless environments it'll hit the fallback, which is fine — actions should never crash the daemon or API just because the display isn't available.

## 3. Build an action registry

**`app/actions/registry.py`:**
```python
from typing import Dict, List
from app.actions.base import Action
from app.actions.reminder import CreateReminderAction
from app.actions.notify import SendNotificationAction

_ACTIONS: Dict[str, Action] = {
    "create_reminder": CreateReminderAction(),
    "send_notification": SendNotificationAction(),
}

def get_action(name: str) -> Action:
    if name not in _ACTIONS:
        raise ValueError(f"Unknown action: {name}")
    return _ACTIONS[name]

def all_tool_specs() -> List[dict]:
    return [action.to_tool_spec() for action in _ACTIONS.values()]
```

Adding a new action later (e.g., `create_calendar_event`) only means writing a new `Action` subclass and registering it here — nothing else needs to change.

---

## 4. Extend `LLMProvider` to support tool calling

The existing `complete()` method (from the reasoning-daemon upgrade) returns plain text. Tool calling needs a method that can return either a text answer or a request to call a tool.

**Update `app/synthesis/base.py`:**
```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class LLMProvider(ABC):
    @abstractmethod
    def synthesize(self, query: str, context_chunks: List[Dict]) -> str:
        ...

    @abstractmethod
    def complete(self, system_prompt: str, user_message: str, temperature: float = 0.2) -> str:
        ...

    @abstractmethod
    def complete_with_tools(
        self,
        system_prompt: str,
        user_message: str,
        tools: List[Dict[str, Any]],
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Returns either:
          {"type": "text", "content": "<answer>"}
        or:
          {"type": "tool_call", "name": "<tool_name>", "arguments": {...}}
        """
        ...
```

**Update `app/synthesis/groq_client.py`** — add:
```python
    def complete_with_tools(self, system_prompt, user_message, tools, temperature=0.2):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            tools=tools,
            tool_choice="auto",
            temperature=temperature,
            max_tokens=800,
        )
        message = response.choices[0].message

        if message.tool_calls:
            call = message.tool_calls[0]  # handle first tool call; extend to loop if you want multi-call support
            import json as _json
            return {
                "type": "tool_call",
                "name": call.function.name,
                "arguments": _json.loads(call.function.arguments),
            }
        return {"type": "text", "content": message.content}
```

---

## 5. Wire tool calling into `/query`

Update the query flow so the LLM can either answer normally (existing RAG synthesis) or decide to take an action instead/in addition.

**Update `app/retrieval/retrieve.py` or `app/main.py`'s `/query` handler:**
```python
from app.actions.registry import all_tool_specs, get_action

QUERY_SYSTEM_PROMPT = """You are a personal memory assistant. You can either answer the \
user's question using the provided memories, or call a tool if the user is asking you to \
remember/remind them of something actionable (e.g., "remind me to X", "don't let me forget Y").

Only call a tool if the user's request is clearly asking for an action to be taken, not just \
information recall. Otherwise, answer directly using the memories provided."""

def handle_query(query: str, top_k: int) -> dict:
    ranked = retrieve_and_rerank(query, top_k)
    llm = get_llm_provider()

    context_text = "\n\n".join(f"[{r['metadata'].get('source', 'unknown')}] {r['text']}" for r in ranked)
    user_message = f"Memories:\n{context_text}\n\nUser request: {query}"

    result = llm.complete_with_tools(
        system_prompt=QUERY_SYSTEM_PROMPT,
        user_message=user_message,
        tools=all_tool_specs(),
    )

    if result["type"] == "tool_call":
        action = get_action(result["name"])
        action_result = action.execute(**result["arguments"])
        return {
            "answer": f"Done — {result['name'].replace('_', ' ')} action taken.",
            "action_taken": result["name"],
            "action_result": action_result,
            "sources": ranked,
        }

    return {
        "answer": result["content"],
        "action_taken": None,
        "action_result": None,
        "sources": ranked,
    }
```

Update `QueryResponse` in `app/models.py` to include the new optional fields:
```python
class QueryResponse(BaseModel):
    answer: str
    action_taken: Optional[str] = None
    action_result: Optional[Dict[str, Any]] = None
    sources: List[QueryResult]
```

---

## 6. Wire actions into the reasoning daemon too

The reasoning daemon (from the previous upgrade) currently only returns one of four fixed actions (`ingest_silent`, `ingest_and_surface`, `ingest_and_flag_conflict`, `skip`). Extend it so that when it detects something with a clear due date or urgency (e.g., clipboard text like "Passport expires September 20 — renew before then"), it can also trigger `create_reminder` automatically.

**Update `app/agent/schema.py`** — add an optional field:
```python
class AgentDecision(BaseModel):
    action: Literal["ingest_silent", "ingest_and_surface", "ingest_and_flag_conflict", "skip"]
    reasoning: str
    related_memory_ids: List[str] = []
    conflict_summary: Optional[str] = None
    suggested_reminder: Optional[str] = None  # new: text for a reminder, if the LLM thinks one is warranted
```

**Update `DECISION_SYSTEM_PROMPT` in `app/agent/reasoner.py`** to mention this new field:
```text
- If the new content mentions a deadline, due date, or something time-sensitive worth \
reminding the user about later, include a short "suggested_reminder" string. Otherwise set \
it to null.
```

**Update `app/daemon.py`** to act on it:
```python
from app.actions.registry import get_action

if decision.suggested_reminder:
    get_action("create_reminder").execute(text=decision.suggested_reminder)
```

This runs regardless of the primary `action` value (silent ingestion can still spawn a reminder if warranted).

---

## 7. New endpoint to list reminders

**Add to `app/main.py`:**
```python
import json
from pathlib import Path

@app.get("/reminders")
def list_reminders():
    path = Path("reminders.json")
    if not path.exists():
        return {"reminders": []}
    return {"reminders": json.loads(path.read_text())}
```

---

## 8. Tests

**`tests/test_actions.py`:**
```python
import json
from pathlib import Path
from app.actions.reminder import CreateReminderAction

def test_create_reminder_persists(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    action = CreateReminderAction()
    result = action.execute(text="Renew passport", due_date="2026-09-01")
    assert result["status"] == "created"

    saved = json.loads(Path("reminders.json").read_text())
    assert len(saved) == 1
    assert saved[0]["text"] == "Renew passport"
```

**`tests/test_query_tool_calling.py`** — mock `complete_with_tools` to return a canned `tool_call` response and assert `/query` executes the action and returns `action_taken` in the response, without needing a live Groq key.

---

## 9. README update

```markdown
## Action-Taking

PAMA is no longer limited to answering questions with text. Through Groq's tool-calling
support, `/query` can create reminders or send desktop notifications when the user's request
is actionable (e.g., "remind me to renew my passport"), rather than only returning a synthesized
answer. The clipboard reasoning daemon also proactively creates reminders when it detects a
deadline or time-sensitive detail in newly captured content, without being asked. Reminders are
listed at `GET /reminders`.
```

---

## Acceptance Criteria

- [ ] `app/actions/base.py`, `reminder.py`, `notify.py`, `registry.py` exist and are wired together.
- [ ] `LLMProvider` has `complete_with_tools()`; `GroqProvider` implements it using Groq's native tool-calling.
- [ ] `/query` can either answer normally or execute a tool call, reflected in the `QueryResponse` model (`action_taken`, `action_result`).
- [ ] `GET /reminders` returns persisted reminders.
- [ ] The reasoning daemon can independently spawn a reminder via `suggested_reminder` even on an `ingest_silent` decision.
- [ ] Notification failures (headless/Docker environments) degrade gracefully to a log line rather than crashing.
- [ ] Tests cover reminder persistence and tool-call handling in `/query` with mocked LLM responses.
- [ ] Manual test: `curl -X POST /query -d '{"query":"remind me to renew my passport before September"}'` results in a new entry in `reminders.json`, not just a text answer.
