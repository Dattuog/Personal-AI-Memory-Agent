# PAMA — Agentic Capabilities: Full Implementation Spec

This is a consolidated, dependency-ordered spec combining four upgrades into one seamless implementation:
1. **Reasoning daemon** (Groq decides what to do with new clipboard content, not fixed similarity rules)
2. **Action-taking** (real tools: reminders, notifications)
3. **Scheduled self-review** (autonomous periodic scan of all memories)
4. **Multi-step tasks** (planner that decomposes complex queries)

These depend on each other (self-review needs actions; the daemon needs actions; multi-step needs the same LLM primitives), so **implement in the phase order below**, not in the order listed above. Each phase should be working and tested before moving to the next.

---

## Phase A — Foundation: extend `LLMProvider` with reasoning + tool-calling primitives

Everything else depends on this. The existing `LLMProvider.synthesize()` is for RAG-style answer synthesis only. Add two more general-purpose methods.

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
        """Raw chat completion, no RAG context injection. Used for reasoning/classification/planning."""
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

**Update `app/synthesis/groq_client.py`** — add both methods to `GroqProvider`:
```python
    def complete(self, system_prompt: str, user_message: str, temperature: float = 0.2) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=800,
        )
        return response.choices[0].message.content

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
            call = message.tool_calls[0]
            import json as _json
            return {
                "type": "tool_call",
                "name": call.function.name,
                "arguments": _json.loads(call.function.arguments),
            }
        return {"type": "text", "content": message.content}
```

**Checkpoint before continuing:** write a throwaway script calling `get_llm_provider().complete(...)` with a trivial prompt and confirm it returns text. Don't proceed to Phase B until this works.

---

## Phase B — Action-taking: give the agent real tools

**`app/actions/base.py`:**
```python
from abc import ABC, abstractmethod
from typing import Any, Dict

class Action(ABC):
    name: str
    description: str
    parameters_schema: Dict[str, Any]

    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        ...

    def to_tool_spec(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }
```

**`app/actions/reminder.py`:**
```python
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional
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

    def execute(self, text: str, due_date: Optional[str] = None) -> Dict[str, Any]:
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
            print(f"[NOTIFY] {title}: {message} (desktop notification unavailable: {e})")
            return {"status": "logged_fallback", "channel": "log"}
```

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

Add `plyer` to `requirements.txt`.

**Checkpoint:** write a throwaway script calling `get_action("create_reminder").execute(text="test")` and confirm `reminders.json` is created correctly. Don't proceed until this works.

---

## Phase C — Reasoning daemon (replaces fixed-threshold clipboard logic)

**`app/agent/schema.py`** — note this schema already includes the `self_review_flag` action and `suggested_reminder` field needed by later phases, so it doesn't need revisiting:
```python
from pydantic import BaseModel
from typing import Literal, List, Optional

class AgentDecision(BaseModel):
    action: Literal[
        "ingest_silent",
        "ingest_and_surface",
        "ingest_and_flag_conflict",
        "skip",
        "self_review_flag",   # used only by Phase D, defined here now to avoid a later schema migration
    ]
    reasoning: str
    related_memory_ids: List[str] = []
    conflict_summary: Optional[str] = None
    suggested_reminder: Optional[str] = None
```

**`app/agent/log_store.py`:**
```python
import json
import time
from pathlib import Path
from typing import List
from app.agent.schema import AgentDecision

LOG_PATH = Path("agent_log.json")

def log_decision(content: str, decision: AgentDecision) -> None:
    entry = {
        "timestamp": time.time(),
        "content": content[:200],
        "action": decision.action,
        "reasoning": decision.reasoning,
        "related_memory_ids": decision.related_memory_ids,
        "conflict_summary": decision.conflict_summary,
    }
    existing = []
    if LOG_PATH.exists():
        existing = json.loads(LOG_PATH.read_text())
    existing.append(entry)
    LOG_PATH.write_text(json.dumps(existing, indent=2))

def get_recent_decisions(n: int = 20) -> List[dict]:
    if not LOG_PATH.exists():
        return []
    existing = json.loads(LOG_PATH.read_text())
    return existing[-n:]
```

**`app/agent/reasoner.py`:**
```python
import json
from typing import List, Dict
from app.agent.schema import AgentDecision
from app.synthesis import get_llm_provider

DECISION_SYSTEM_PROMPT = """You are the reasoning layer of a personal memory agent. \
You are given a new piece of text the user just copied, along with a list of existing \
memories that are semantically related to it (if any).

Decide what action to take. Respond with ONLY a JSON object matching this exact schema, \
no other text, no markdown fences:

{
  "action": "ingest_silent" | "ingest_and_surface" | "ingest_and_flag_conflict" | "skip",
  "reasoning": "<one or two sentence explanation>",
  "related_memory_ids": ["<id>", ...],
  "conflict_summary": "<string or null>",
  "suggested_reminder": "<string or null>"
}

Guidelines:
- Use "skip" if the new content is noise, gibberish, or clearly not worth remembering.
- Use "ingest_and_flag_conflict" if the new content contradicts or supersedes an existing \
memory (e.g., a new date/decision that replaces an old one on the same topic).
- Use "ingest_and_surface" if the new content is meaningfully related to one or more \
existing memories and the user would likely want to be reminded of that connection.
- Use "ingest_silent" for anything else worth storing but not otherwise noteworthy.
- If the new content mentions a deadline, due date, or something time-sensitive worth \
reminding the user about later, set "suggested_reminder" to a short reminder string. \
Otherwise set it to null. This can be set regardless of which action you choose.
- Only include memory IDs that are genuinely relevant in related_memory_ids.
- Never output the "self_review_flag" action here — that value is reserved for a different \
part of the system.
"""

def _build_user_message(new_content: str, candidates: List[Dict]) -> str:
    if not candidates:
        candidate_block = "No related memories were found."
    else:
        lines = []
        for c in candidates:
            lines.append(f"id: {c['id']}\ntext: {c['text']}\ntimestamp: {c['metadata'].get('timestamp')}")
        candidate_block = "\n\n".join(lines)
    return f"New clipboard content:\n{new_content}\n\nRelated existing memories:\n{candidate_block}"

def decide(new_content: str, candidates: List[Dict]) -> AgentDecision:
    llm = get_llm_provider()
    user_message = _build_user_message(new_content, candidates)
    raw = llm.complete(system_prompt=DECISION_SYSTEM_PROMPT, user_message=user_message, temperature=0.1)
    try:
        parsed = json.loads(raw)
        return AgentDecision(**parsed)
    except Exception:
        return AgentDecision(
            action="ingest_silent",
            reasoning="Fallback: could not parse LLM decision.",
            related_memory_ids=[],
        )
```

**Update `app/daemon.py`** — replace fixed-threshold surfacing with reasoning + action-taking:
```python
import asyncio
import hashlib
import time
import pyperclip

from app.embeddings import get_embedding_provider
from app.storage import get_vector_store
from app.security.sensitive_filter import looks_sensitive
from app.agent.reasoner import decide
from app.agent.log_store import log_decision
from app.actions.registry import get_action
from app.config import settings

async def clipboard_watcher(poll_interval: float = 1.0):
    last_hash = None
    while True:
        try:
            content = pyperclip.paste()
        except Exception:
            content = None

        if content and content.strip():
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            if content_hash != last_hash:
                last_hash = content_hash
                await handle_new_clipboard_content(content)

        await asyncio.sleep(poll_interval)


async def handle_new_clipboard_content(content: str):
    if len(content.strip()) < 20:
        return
    if looks_sensitive(content):
        return

    embedding_provider = get_embedding_provider()
    store = get_vector_store()
    embedding = embedding_provider.embed(content)

    # near-duplicate check against existing memories
    near_dupes = store.query(embedding, k=3)
    if near_dupes and (1 - near_dupes[0]["distance"]) > 0.95:
        return

    candidates = store.query(embedding, k=10)
    candidates = [c for c in candidates if (1 - c["distance"]) >= 0.15]

    decision = decide(content, candidates)
    log_decision(content, decision)

    if decision.action == "skip":
        return

    timestamp = time.time()
    new_id = f"clipboard-{timestamp}"
    store.add(new_id, embedding, content, {"source": "clipboard", "timestamp": timestamp})

    if decision.suggested_reminder:
        get_action("create_reminder").execute(text=decision.suggested_reminder)

    if decision.action == "ingest_and_surface":
        related_texts = [c["text"] for c in candidates if c["id"] in decision.related_memory_ids]
        print(f"[SURFACED] {content[:60]} relates to: {related_texts}")

    elif decision.action == "ingest_and_flag_conflict":
        print(f"[CONFLICT] {content[:60]} — {decision.conflict_summary}")


async def main():
    await asyncio.gather(
        clipboard_watcher(),
        self_review_loop(),  # added in Phase D — if implementing Phase C standalone first, comment this out temporarily
    )

if __name__ == "__main__":
    asyncio.run(main())
```

Note: `self_review_loop` is defined in Phase D below. If you're testing Phase C in isolation first, temporarily run just `clipboard_watcher()` in `main()` and wire in `self_review_loop` once Phase D is implemented.

**Checkpoint:** ingest a memory (e.g., "Passport renewed in January"), then copy a contradicting clipboard string (e.g., "Actually renewed passport in June instead") and confirm `agent_log.json` shows `ingest_and_flag_conflict`. Don't proceed until this works.

---

## Phase D — Scheduled self-review (autonomous periodic scan)

**Update `app/storage/base.py`** — add two methods to the interface:
```python
    @abstractmethod
    def list_all(self, limit: int = 1000) -> List[Dict[str, Any]]:
        ...

    @abstractmethod
    def update_metadata(self, id: str, metadata: Dict[str, Any]) -> None:
        ...
```

**Implement in `app/storage/chroma_store.py`:**
```python
    def list_all(self, limit: int = 1000) -> list[dict]:
        result = self.collection.get(limit=limit)
        rows = []
        ids = result.get("ids", [])
        docs = result.get("documents", [])
        metadatas = result.get("metadatas", [])
        for item_id, text, metadata in zip(ids, docs, metadatas):
            rows.append({"id": item_id, "text": text, "metadata": metadata or {}})
        return rows

    def update_metadata(self, id: str, metadata: dict) -> None:
        existing = self.collection.get(ids=[id])
        current_meta = (existing.get("metadatas") or [{}])[0] or {}
        current_meta.update(metadata)
        self.collection.update(ids=[id], metadatas=[current_meta])
```

**Implement (or stub with `NotImplementedError`, documented in README) in `app/storage/pinecone_store.py`.**

**Update `app/config.py`** — add:
```python
    self_review_interval_hours: float = 24.0
    self_review_cooldown_hours: float = 48.0
```

**Update `.env.example`:**
```text
SELF_REVIEW_INTERVAL_HOURS=24
SELF_REVIEW_COOLDOWN_HOURS=48
```

**`app/agent/self_review.py`:**
```python
import json
import time
from datetime import datetime
from typing import List, Dict
from app.synthesis import get_llm_provider
from app.storage import get_vector_store
from app.actions.registry import get_action
from app.agent.log_store import log_decision
from app.agent.schema import AgentDecision
from app.config import settings

SELF_REVIEW_SYSTEM_PROMPT = """You are the self-review layer of a personal memory agent. \
You are given a batch of stored memories along with today's date. Review them for anything \
time-sensitive: upcoming deadlines, unresolved tasks, mentioned dates that are approaching, \
or things the user said they needed to do but likely haven't confirmed doing.

Respond with ONLY a JSON array (no other text), where each element has this shape:
{
  "memory_id": "<id>",
  "needs_attention": true | false,
  "reminder_text": "<short reminder text, or null if needs_attention is false>",
  "reasoning": "<one sentence>"
}

Only mark needs_attention as true for genuinely time-sensitive items."""


def _build_batch_message(memories: List[Dict]) -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    lines = [f"Today's date: {today}", "", "Memories:"]
    for m in memories:
        ts = m["metadata"].get("timestamp")
        date_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d") if ts else "unknown"
        lines.append(f"id: {m['id']} | stored: {date_str}\ntext: {m['text']}")
    return "\n\n".join(lines)


def run_self_review(batch_size: int = 25) -> List[Dict]:
    store = get_vector_store()
    llm = get_llm_provider()
    all_memories = store.list_all(limit=1000)

    now = time.time()
    cooldown_seconds = settings.self_review_cooldown_hours * 3600
    candidates = [
        m for m in all_memories
        if now - float(m["metadata"].get("last_reviewed_at", 0)) > cooldown_seconds
    ]

    results = []
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i + batch_size]
        message = _build_batch_message(batch)
        raw = llm.complete(system_prompt=SELF_REVIEW_SYSTEM_PROMPT, user_message=message, temperature=0.1)

        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = []

        for item in parsed:
            memory_id = item.get("memory_id")
            if not memory_id:
                continue

            store.update_metadata(memory_id, {"last_reviewed_at": now})

            if item.get("needs_attention") and item.get("reminder_text"):
                get_action("create_reminder").execute(text=item["reminder_text"])
                decision = AgentDecision(
                    action="self_review_flag",
                    reasoning=item.get("reasoning", ""),
                    related_memory_ids=[memory_id],
                )
                log_decision(item["reminder_text"], decision)
                results.append({"memory_id": memory_id, "reminder_text": item["reminder_text"]})

    return results
```

**Add the scheduler loop to `app/daemon.py`** (referenced in Phase C's `main()`):
```python
from app.agent.self_review import run_self_review

async def self_review_loop():
    while True:
        try:
            flagged = run_self_review()
            print(f"[SELF-REVIEW] Completed. {len(flagged)} items flagged.")
        except Exception as e:
            print(f"[SELF-REVIEW] Error: {e}")
        await asyncio.sleep(settings.self_review_interval_hours * 3600)
```

**Add a manual trigger endpoint to `app/main.py`:**
```python
from app.agent.self_review import run_self_review

@app.post("/self-review/run")
def trigger_self_review():
    flagged = run_self_review()
    return {"flagged_count": len(flagged), "flagged": flagged}
```

**Add a reminders listing endpoint:**
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

**Checkpoint:** ingest a memory like "Passport expires October 2026, renew before then," call `POST /self-review/run`, and confirm a new entry appears in `reminders.json`. Don't proceed until this works.

---

## Phase E — Multi-step tasks (query planner)

**`app/agent/planner.py`:**
```python
import json
from typing import List, Dict, Any
from app.synthesis import get_llm_provider
from app.retrieval.retrieve import retrieve_and_rerank

CLASSIFY_SYSTEM_PROMPT = """Classify the user's request as either "simple" or "complex".

"simple" = a single factual lookup answerable from one retrieval pass.
"complex" = a multi-part request that benefits from being broken into sub-questions.

Respond with ONLY one word: simple or complex."""

def classify_query(query: str) -> str:
    llm = get_llm_provider()
    raw = llm.complete(system_prompt=CLASSIFY_SYSTEM_PROMPT, user_message=query, temperature=0.0)
    return "complex" if "complex" in raw.strip().lower() else "simple"


PLAN_SYSTEM_PROMPT = """You are the planning layer of a personal memory agent. Break the \
user's request into 2-5 concrete sub-questions that, together, cover what's needed to fully \
address it. Each sub-question should be answerable by searching the user's personal memory \
store.

Respond with ONLY a JSON array of strings, no other text:
["sub-question 1", "sub-question 2", ...]"""

def build_plan(query: str) -> List[str]:
    llm = get_llm_provider()
    raw = llm.complete(system_prompt=PLAN_SYSTEM_PROMPT, user_message=query, temperature=0.2)
    try:
        steps = json.loads(raw)
        if isinstance(steps, list) and all(isinstance(s, str) for s in steps):
            return steps[:5]
    except Exception:
        pass
    return [query]


def execute_plan(steps: List[str], top_k_per_step: int = 3) -> List[Dict[str, Any]]:
    step_results = []
    for step in steps:
        ranked = retrieve_and_rerank(step, top_k_per_step)
        step_results.append({
            "sub_question": step,
            "found_memories": ranked,
            "has_data": len(ranked) > 0,
        })
    return step_results


FINAL_SYNTHESIS_SYSTEM_PROMPT = """You are a personal memory assistant. You were given a \
multi-part task, broken into sub-questions, each with whatever memories were found (or none). \
Write a clear final answer that:
1. Answers what you can from the memories found, organized by sub-topic.
2. Explicitly lists which sub-questions had NO relevant memories, under a "Still missing"
   section, phrased as things the user hasn't told you yet.
Keep it concise. Do not invent information not present in the memories."""

def _format_step_results(step_results: List[Dict[str, Any]]) -> str:
    blocks = []
    for step in step_results:
        if step["has_data"]:
            mem_lines = "\n".join(f"- {m['text']}" for m in step["found_memories"])
        else:
            mem_lines = "(no relevant memories found)"
        blocks.append(f"Sub-question: {step['sub_question']}\n{mem_lines}")
    return "\n\n".join(blocks)

def synthesize_plan_answer(original_query: str, step_results: List[Dict[str, Any]]) -> str:
    llm = get_llm_provider()
    context = _format_step_results(step_results)
    user_message = f"Original request: {original_query}\n\n{context}"
    return llm.complete(system_prompt=FINAL_SYNTHESIS_SYSTEM_PROMPT, user_message=user_message, temperature=0.3)


def run_multi_step_task(query: str) -> Dict[str, Any]:
    steps = build_plan(query)
    step_results = execute_plan(steps)
    answer = synthesize_plan_answer(query, step_results)
    missing = [s["sub_question"] for s in step_results if not s["has_data"]]
    all_sources = [m for s in step_results for m in s["found_memories"]]
    return {"answer": answer, "plan": steps, "missing_info": missing, "sources": all_sources}
```

---

## Phase F — Final `/query` integration (ties Phase B + E together)

This is the one place all four capabilities meet: classify → complex goes to the planner (Phase E), simple goes to the existing retrieval path, which itself can either answer directly or call a tool (Phase B).

**Update `app/models.py`:**
```python
from typing import Optional, Dict, Any, List

class QueryResponse(BaseModel):
    answer: str
    action_taken: Optional[str] = None
    action_result: Optional[Dict[str, Any]] = None
    sources: List[QueryResult]
    plan: Optional[List[str]] = None
    missing_info: Optional[List[str]] = None
```

**Update `app/main.py`:**
```python
from app.agent.planner import classify_query, run_multi_step_task
from app.actions.registry import all_tool_specs, get_action
from app.retrieval.retrieve import retrieve_and_rerank
from app.synthesis import get_llm_provider

SIMPLE_QUERY_SYSTEM_PROMPT = """You are a personal memory assistant. You can either answer the \
user's question using the provided memories, or call a tool if the user is asking you to \
remember/remind them of something actionable (e.g., "remind me to X").

Only call a tool if the request clearly asks for an action. Otherwise, answer directly using \
the memories provided."""

def handle_simple_query(query: str, top_k: int) -> QueryResponse:
    ranked = retrieve_and_rerank(query, top_k)
    llm = get_llm_provider()
    context_text = "\n\n".join(f"[{r['metadata'].get('source', 'unknown')}] {r['text']}" for r in ranked)
    user_message = f"Memories:\n{context_text}\n\nUser request: {query}"

    result = llm.complete_with_tools(
        system_prompt=SIMPLE_QUERY_SYSTEM_PROMPT,
        user_message=user_message,
        tools=all_tool_specs(),
    )

    if result["type"] == "tool_call":
        action = get_action(result["name"])
        action_result = action.execute(**result["arguments"])
        return QueryResponse(
            answer=f"Done — {result['name'].replace('_', ' ')} action taken.",
            action_taken=result["name"],
            action_result=action_result,
            sources=ranked,
        )

    return QueryResponse(
        answer=result["content"],
        action_taken=None,
        action_result=None,
        sources=ranked,
    )


@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    complexity = classify_query(request.query)

    if complexity == "complex":
        result = run_multi_step_task(request.query)
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            plan=result["plan"],
            missing_info=result["missing_info"],
        )

    return handle_simple_query(request.query, request.top_k)
```

---

## Phase G — Tests

Create these test files (all mock the LLM/store — no live Groq key needed to run the suite):

- `tests/test_reasoner.py` — valid decision parsing, malformed-JSON fallback.
- `tests/test_actions.py` — `CreateReminderAction` persists to `reminders.json` correctly (use `tmp_path` + `monkeypatch.chdir`).
- `tests/test_self_review.py` — mocked batch review flags an item, creates a reminder, updates metadata.
- `tests/test_planner.py` — classification, plan parsing + fallback, gap detection in `execute_plan`.
- `tests/test_query_integration.py` — mock `classify_query` to return "simple"/"complex" and confirm `/query` routes correctly in both cases; mock `complete_with_tools` to return a tool call and confirm the action executes and `action_taken` is populated in the response.

Run everything with:
```bash
pytest -v
```

---

## Phase H — README consolidation

Replace/add these sections in `README.md`:

```markdown
## Agentic Capabilities

PAMA goes beyond passive storage-and-retrieval in four ways:

- **Reasoning daemon** — the clipboard watcher no longer relies on a fixed similarity
  threshold. New clipboard content, along with related existing memories, is handed to Groq,
  which decides whether to store it silently, surface a meaningful connection, flag a
  conflict with existing information, or skip it as noise.
- **Action-taking** — through Groq's native tool-calling, `/query` can create reminders or
  send desktop notifications when a request is actionable (e.g., "remind me to renew my
  passport"), instead of only returning text. The reasoning daemon can also spawn reminders
  on its own when it detects a deadline in newly captured content.
- **Scheduled self-review** — independent of any query or clipboard event, PAMA periodically
  (default every 24 hours) reviews its entire memory store for time-sensitive items and
  creates reminders proactively. Reviewed memories are cooled down for 48 hours by default to
  avoid repeat nagging. Trigger on demand with `POST /self-review/run`.
- **Multi-step tasks** — complex requests (e.g., "help me plan my September trip") are broken
  into sub-questions, each retrieved independently, with the final answer explicitly
  separating what was found from what's still missing, rather than guessing.

Reminders are listed at `GET /reminders`. Reasoning/self-review decisions are logged to
`agent_log.json`.
```

---

## Build Order Summary

1. Phase A (LLM primitives) — checkpoint before continuing.
2. Phase B (actions) — checkpoint before continuing.
3. Phase C (reasoning daemon) — checkpoint before continuing.
4. Phase D (self-review) — checkpoint before continuing.
5. Phase E (planner) — no external side effects, safe to build without a checkpoint.
6. Phase F (final `/query` wiring) — this is where everything becomes user-facing; test thoroughly here.
7. Phase G (tests) — can be written incrementally alongside each phase instead of all at the end, if preferred.
8. Phase H (README).

## Master Acceptance Criteria

- [ ] `LLMProvider` has `complete()` and `complete_with_tools()`; `GroqProvider` implements both.
- [ ] Action registry supports `create_reminder` and `send_notification`, both degrade gracefully on failure.
- [ ] Clipboard daemon uses `decide()` (LLM reasoning) instead of a fixed similarity threshold, and acts on all four decision types plus `suggested_reminder`.
- [ ] `VectorStore` has `list_all()` and `update_metadata()` (Chroma implemented; Pinecone implemented or explicitly `NotImplementedError` + documented).
- [ ] Self-review runs on a schedule, respects the cooldown window, and is triggerable on demand via `POST /self-review/run`.
- [ ] `/query` classifies simple vs. complex and routes accordingly; simple queries can trigger tool calls; complex queries return a plan + missing-info breakdown.
- [ ] All new modules have tests using mocked LLM/store calls — the test suite should not require a live Groq key to pass.
- [ ] End-to-end manual test: ingest partial trip info (flights only), query "help me plan my September trip" → confirm missing hotel/visa info is flagged; then query "remind me to book a hotel" → confirm a reminder is created; then call `POST /self-review/run` → confirm no duplicate reminder is spawned for something already reviewed within the cooldown window.
