# PAMA — Upgrade: Scheduled Self-Review

## Goal

So far, everything PAMA does is triggered by something external — a `/query` call or a clipboard paste. This upgrade adds genuine autonomy: a background job that runs on its own schedule (e.g., once a day), reviews the entire memory store without being asked, and proactively creates reminders/notifications for anything time-sensitive it finds — using the same reasoning and action-taking infrastructure already built.

This depends on the two previous upgrades already being implemented:
- **Reasoning daemon** (`app/agent/reasoner.py`, `app/agent/schema.py`, `app/synthesis` `complete()`)
- **Action-taking** (`app/actions/registry.py`, `create_reminder`, `send_notification`)

---

## 1. Add a "list all memories" capability to the storage layer

The self-review job needs to scan the whole store, not just query by similarity. Add this to the `VectorStore` interface.

**Update `app/storage/base.py`:**
```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class VectorStore(ABC):
    @abstractmethod
    def add(self, id: str, embedding: List[float], text: str, metadata: Dict[str, Any]) -> None:
        ...

    @abstractmethod
    def query(self, embedding: List[float], k: int) -> List[Dict[str, Any]]:
        ...

    @abstractmethod
    def delete(self, id: str) -> None:
        ...

    @abstractmethod
    def list_all(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Return all stored memories (id, text, metadata), for scans that aren't query-driven."""
        ...

    @abstractmethod
    def update_metadata(self, id: str, metadata: Dict[str, Any]) -> None:
        """Merge/overwrite metadata fields for a given id — used to mark items as reviewed."""
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

**Implement the same two methods in `app/storage/pinecone_store.py`** using `index.query` with a zero/dummy vector and `top_k` for listing (Pinecone doesn't have a clean "list all" — use metadata filtering or fetch by ID range depending on how you're indexing) and `index.update` for metadata. If this is impractical for your Pinecone setup, it's acceptable to document that `list_all` / self-review is currently Chroma-only and raise `NotImplementedError` in `PineconeStore` for now — note this clearly in the README rather than silently failing.

---

## 2. Track review state so memories aren't re-flagged forever

Add a metadata field, `last_reviewed_at`, set whenever the self-review job processes a memory. This prevents the same "renew your passport" memory from generating a new reminder every single day forever.

No new file needed — this is just a convention used by the self-review job itself (step 4).

---

## 3. Build the self-review prompt and logic

**`app/agent/self_review.py`:**
```python
import json
import time
from typing import List, Dict
from app.synthesis import get_llm_provider
from app.storage import get_vector_store
from app.actions.registry import get_action
from app.agent.log_store import log_decision  # reuse existing logger from reasoning daemon
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

Only mark needs_attention as true for genuinely time-sensitive items. Do not flag memories \
that are just general facts, preferences, or things with no clear deadline."""


def _build_batch_message(memories: List[Dict]) -> str:
    from datetime import datetime
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

    # Skip memories reviewed in the last N hours to avoid re-flagging every run
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
            parsed = []  # skip this batch on malformed output rather than crashing the whole run

        for item in parsed:
            memory_id = item.get("memory_id")
            if not memory_id:
                continue

            store.update_metadata(memory_id, {"last_reviewed_at": now})

            if item.get("needs_attention") and item.get("reminder_text"):
                get_action("create_reminder").execute(text=item["reminder_text"])
                log_decision_entry = {
                    "action": "self_review_reminder",
                    "reasoning": item.get("reasoning", ""),
                    "related_memory_ids": [memory_id],
                    "conflict_summary": None,
                }
                results.append(log_decision_entry)

    return results
```

Note: `log_decision` from the reasoning daemon expects an `AgentDecision`-like object — either reuse `AgentDecision` directly here (constructing one with `action="ingest_and_surface"` or a new literal value like `"self_review_flag"` added to the schema), or write these as plain dicts to a separate log file. Simplest path: extend `AgentDecision`'s `action` literal in `app/agent/schema.py` to include `"self_review_flag"`, and construct proper `AgentDecision` objects here so the existing `log_decision()` and `/surfaced` endpoint work without changes.

---

## 4. Add config for the schedule

**Update `app/config.py`:**
```python
    self_review_interval_hours: float = 24.0
    self_review_cooldown_hours: float = 48.0
```

**Update `.env.example`:**
```text
SELF_REVIEW_INTERVAL_HOURS=24
SELF_REVIEW_COOLDOWN_HOURS=48
```

`SELF_REVIEW_INTERVAL_HOURS` controls how often the job runs; `SELF_REVIEW_COOLDOWN_HOURS` controls how long a memory is left alone after being reviewed once, so the agent doesn't nag about the same thing every single day.

---

## 5. Run it as a scheduled background loop

Two options — pick based on how you're already running the daemon.

**Option A — fold into the existing daemon process** (simplest, reuses the process you already have running):

**Update `app/daemon.py`** to run both the clipboard watcher and a self-review loop concurrently:
```python
import asyncio
from app.agent.self_review import run_self_review
from app.config import settings

async def self_review_loop():
    while True:
        try:
            flagged = run_self_review()
            print(f"[SELF-REVIEW] Completed. {len(flagged)} items flagged.")
        except Exception as e:
            print(f"[SELF-REVIEW] Error: {e}")
        await asyncio.sleep(settings.self_review_interval_hours * 3600)

async def main():
    await asyncio.gather(
        clipboard_watcher(),      # existing function
        self_review_loop(),       # new
    )

if __name__ == "__main__":
    asyncio.run(main())
```

**Option B — separate process/container**, if you want it decoupled from clipboard watching (e.g., so it can run on a server without clipboard access at all):

**`app/scheduler.py`** (new file, standalone):
```python
import asyncio
from app.agent.self_review import run_self_review
from app.config import settings

async def main():
    while True:
        try:
            flagged = run_self_review()
            print(f"[SELF-REVIEW] Completed. {len(flagged)} items flagged.")
        except Exception as e:
            print(f"[SELF-REVIEW] Error: {e}")
        await asyncio.sleep(settings.self_review_interval_hours * 3600)

if __name__ == "__main__":
    asyncio.run(main())
```
Run with `python -m app.scheduler`, and add it as its own service in `docker-compose.yml` if using Docker, sharing the same `chroma_data` volume.

**Recommendation:** use Option A for now — it's less to manage and the self-review job is lightweight enough to share a process with the clipboard watcher. Switch to Option B later only if you need independent scaling/restart behavior.

---

## 6. Manual trigger endpoint (important for testing — don't wait 24 hours to verify this works)

**Add to `app/main.py`:**
```python
from app.agent.self_review import run_self_review

@app.post("/self-review/run")
def trigger_self_review():
    flagged = run_self_review()
    return {"flagged_count": len(flagged), "flagged": flagged}
```

This lets you test the feature on demand instead of waiting for the scheduled interval.

---

## 7. Tests

**`tests/test_self_review.py`:**
```python
import json
from unittest.mock import patch, MagicMock

def test_self_review_flags_and_creates_reminder():
    canned_response = json.dumps([
        {
            "memory_id": "mem-1",
            "needs_attention": True,
            "reminder_text": "Renew passport before September",
            "reasoning": "Deadline approaching",
        }
    ])

    fake_memory = {"id": "mem-1", "text": "Renew passport before September", "metadata": {"timestamp": 0}}

    with patch("app.agent.self_review.get_vector_store") as mock_store, \
         patch("app.agent.self_review.get_llm_provider") as mock_llm, \
         patch("app.agent.self_review.get_action") as mock_action:

        mock_store.return_value.list_all.return_value = [fake_memory]
        mock_llm.return_value.complete.return_value = canned_response
        mock_action.return_value.execute = MagicMock()

        from app.agent.self_review import run_self_review
        results = run_self_review()

        assert len(results) == 1
        mock_action.return_value.execute.assert_called_once()
        mock_store.return_value.update_metadata.assert_called_once()
```

---

## 8. README update

```markdown
## Scheduled Self-Review

Beyond reacting to queries and clipboard events, PAMA periodically reviews its entire memory
store on its own (default: every 24 hours) looking for time-sensitive items — approaching
deadlines, unresolved tasks, dates worth acting on. Anything flagged automatically creates a
reminder via the same action-taking system used elsewhere, without the user asking. Reviewed
memories are cooled down for 48 hours by default so the same item isn't re-flagged daily.
Trigger a review on demand with `POST /self-review/run` instead of waiting for the schedule.
```

---

## Acceptance Criteria

- [ ] `VectorStore` interface has `list_all()` and `update_metadata()`; implemented in `ChromaStore` (Pinecone can raise `NotImplementedError` if impractical, documented in README).
- [ ] `app/agent/self_review.py` exists, batches memories, calls the LLM once per batch (not once per memory — avoid excessive API calls), and only flags genuinely time-sensitive items.
- [ ] `last_reviewed_at` metadata prevents the same memory from being re-flagged within the cooldown window.
- [ ] Self-review runs on a schedule (`SELF_REVIEW_INTERVAL_HOURS`) via the daemon process or a separate scheduler process.
- [ ] `POST /self-review/run` allows on-demand triggering for testing.
- [ ] Flagged items create real reminders via the existing `create_reminder` action, and are logged via the existing `log_decision`/`/surfaced` mechanism (reuse `AgentDecision`, extending its `action` literal with `"self_review_flag"`).
- [ ] Test with mocked LLM/store confirms flagging → reminder creation → metadata update all happen correctly.
- [ ] Manual test: ingest a memory like "Passport expires October 2026, renew before then," call `POST /self-review/run`, and confirm a new entry appears in `reminders.json` without you having queried or pasted anything related.
