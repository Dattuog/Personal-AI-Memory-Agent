# PAMA — Upgrade: Reasoning Daemon (Groq-Driven Decision Making)

## Goal

Currently, the clipboard daemon uses a fixed rule: `if similarity > threshold: surface it`. This upgrade replaces that fixed rule with an LLM reasoning step — the daemon hands the new clipboard content + candidate related memories to Groq, and Groq **decides** what action to take, instead of a hardcoded similarity check driving behavior. This is what turns the daemon from "a similarity trigger" into "an agent that reasons about what matters."

---

## 1. Define the decision schema

The daemon needs the LLM to return a structured decision, not free text — so define a strict schema.

**`app/agent/schema.py`:**
```python
from pydantic import BaseModel
from typing import Literal, List, Optional

class RelatedMemoryRef(BaseModel):
    id: str
    reason: str  # why this memory is related, per the LLM's reasoning

class AgentDecision(BaseModel):
    action: Literal["ingest_silent", "ingest_and_surface", "ingest_and_flag_conflict", "skip"]
    reasoning: str                          # short explanation, for logging/debugging
    related_memory_ids: List[str] = []      # memories worth surfacing/flagging, if any
    conflict_summary: Optional[str] = None  # filled only if action == ingest_and_flag_conflict
```

Action meanings:
- `ingest_silent` — store it, nothing else is noteworthy (e.g., a random unrelated note).
- `ingest_and_surface` — store it, and this connects meaningfully to existing memories the user should be reminded of.
- `ingest_and_flag_conflict` — store it, but it contradicts or supersedes an existing memory (e.g., two different dates for the same event) — needs the user's attention.
- `skip` — don't store this at all (e.g., it's noise, or a near-duplicate already covered).

---

## 2. Build the reasoning module

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
  "conflict_summary": "<string or null>"
}

Guidelines:
- Use "skip" if the new content is noise, gibberish, or clearly not worth remembering \
(e.g., a copied password prompt, a random UI string, an empty-feeling fragment).
- Use "ingest_and_flag_conflict" if the new content contradicts or supersedes an existing \
memory (e.g., a new date/decision that replaces an old one on the same topic).
- Use "ingest_and_surface" if the new content is meaningfully related to one or more \
existing memories and the user would likely want to be reminded of that connection.
- Use "ingest_silent" for anything else worth storing but not otherwise noteworthy.
- Only include memory IDs that are genuinely relevant in related_memory_ids.
"""

def _build_user_message(new_content: str, candidates: List[Dict]) -> str:
    if not candidates:
        candidate_block = "No related memories were found."
    else:
        lines = []
        for c in candidates:
            lines.append(f"id: {c['id']}\ntext: {c['text']}\ntimestamp: {c['metadata'].get('timestamp')}")
        candidate_block = "\n\n".join(lines)

    return (
        f"New clipboard content:\n{new_content}\n\n"
        f"Related existing memories:\n{candidate_block}"
    )

def decide(new_content: str, candidates: List[Dict]) -> AgentDecision:
    llm = get_llm_provider()
    user_message = _build_user_message(new_content, candidates)

    raw = llm.synthesize(
        query=user_message,
        context_chunks=[],  # not used here; see note below on provider interface
    )
    # NOTE: see step 3 — LLMProvider needs a raw-completion method, not just RAG synthesis
    try:
        parsed = json.loads(raw)
        return AgentDecision(**parsed)
    except Exception:
        # fail safe: fall back to the old behavior if the LLM response is malformed
        return AgentDecision(
            action="ingest_silent",
            reasoning="Fallback: could not parse LLM decision.",
            related_memory_ids=[],
        )
```

## 3. Extend `LLMProvider` with a raw-completion method

The existing `LLMProvider.synthesize()` is built for RAG-style answer synthesis (query + context chunks → grounded answer). The reasoning step needs a more general "give me a raw completion for this system+user prompt" method. Add this to the interface rather than hacking `synthesize()` for a different purpose.

**Update `app/synthesis/base.py`:**
```python
from abc import ABC, abstractmethod
from typing import List, Dict

class LLMProvider(ABC):
    @abstractmethod
    def synthesize(self, query: str, context_chunks: List[Dict]) -> str:
        ...

    @abstractmethod
    def complete(self, system_prompt: str, user_message: str, temperature: float = 0.2) -> str:
        """Raw chat completion, no RAG context injection. Used for agent reasoning steps."""
        ...
```

**Update `app/synthesis/groq_client.py`** — add:
```python
    def complete(self, system_prompt: str, user_message: str, temperature: float = 0.2) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=500,
        )
        return response.choices[0].message.content
```

**Fix `reasoner.py`'s `decide()` to use `complete()` instead of `synthesize()`:**
```python
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

Low temperature (0.1) keeps the JSON output more consistent and less prone to drifting off-schema.

---

## 4. Wire the reasoner into the daemon

**Update `app/daemon.py`** — replace the fixed-threshold logic with a call to `decide()`:

```python
from app.agent.reasoner import decide
from app.agent.schema import AgentDecision

async def process_clipboard_entry(content: str, embedding):
    store = get_vector_store()

    # Get candidate related memories BEFORE deciding whether/how to ingest
    candidates = store.query(embedding, k=10)
    candidates = [c for c in candidates if (1 - c["distance"]) >= 0.15]  # loose pre-filter, final call is the LLM's

    decision: AgentDecision = decide(content, candidates)
    log_decision(content, decision)  # simple print/log for now, see step 5

    if decision.action == "skip":
        return

    # All remaining actions ingest the content
    timestamp = time.time()
    new_id = store_new_memory(content, embedding, timestamp)  # wraps ingest_entry logic

    if decision.action == "ingest_and_surface":
        await surface_to_user(content, decision.related_memory_ids, decision.reasoning)

    elif decision.action == "ingest_and_flag_conflict":
        await flag_conflict(content, decision.related_memory_ids, decision.conflict_summary)

    # ingest_and_surface / ingest_and_flag_conflict / ingest_silent all fall through here after their branch
```

Keep the existing guardrails (sensitive-content filter, min-length check, exact-duplicate hash check) running **before** this — no need to burn an LLM call on a copied password or a repeat paste. Those checks stay as fast, free, local pre-filters; the reasoning step only runs on things that pass them.

---

## 5. Surfaced/flagged storage — extend beyond simple similarity records

Your existing `/surfaced` endpoint likely just lists similarity-matched memories. Extend the storage record to include the reasoning and distinguish surface vs. conflict:

**`app/agent/log_store.py`** (new, simple JSON or SQLite-backed store):
```python
import json
import time
from pathlib import Path
from typing import List

LOG_PATH = Path("agent_log.json")

def log_decision(content: str, decision) -> None:
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

**Update `/surfaced` endpoint in `app/main.py`** to read from `get_recent_decisions()` and optionally filter by `action != "ingest_silent"` so the user only sees the notable ones (surfaced connections + conflicts), not every silent ingestion.

---

## 6. Testing this without burning API calls constantly

Add a test that mocks `get_llm_provider().complete` to return canned JSON strings representing each of the four actions, and asserts the daemon branches correctly:

**`tests/test_reasoner.py`:**
```python
from unittest.mock import patch
from app.agent.reasoner import decide

def test_decide_parses_valid_json():
    canned = '{"action": "ingest_and_surface", "reasoning": "test", "related_memory_ids": ["abc"], "conflict_summary": null}'
    with patch("app.agent.reasoner.get_llm_provider") as mock_llm:
        mock_llm.return_value.complete.return_value = canned
        result = decide("some clipboard text", [])
        assert result.action == "ingest_and_surface"
        assert result.related_memory_ids == ["abc"]

def test_decide_falls_back_on_malformed_json():
    with patch("app.agent.reasoner.get_llm_provider") as mock_llm:
        mock_llm.return_value.complete.return_value = "not valid json at all"
        result = decide("some clipboard text", [])
        assert result.action == "ingest_silent"
        assert "Fallback" in result.reasoning
```

---

## 7. README update

Add a short section:

```markdown
## Reasoning Agent (Clipboard Daemon)

The clipboard daemon no longer relies on a fixed similarity threshold to decide what to
surface. New clipboard content, along with any semantically related existing memories, is
handed to Groq, which decides one of four actions: silently store it, store and surface a
meaningful connection, store and flag a conflict with existing information, or skip it
entirely as noise. This reasoning step is logged to `agent_log.json` with the model's stated
reasoning for each decision, which the `/surfaced` endpoint exposes (surfaced connections and
flagged conflicts only, silent ingestions are hidden by default).
```

---

## Acceptance Criteria

- [ ] `LLMProvider` has a new `complete()` method alongside `synthesize()`; `GroqProvider` implements it.
- [ ] `app/agent/schema.py`, `app/agent/reasoner.py`, `app/agent/log_store.py` exist.
- [ ] Clipboard daemon calls `decide()` after the existing fast local pre-filters (length/secret/dedupe checks), not instead of them.
- [ ] All four decision actions (`ingest_silent`, `ingest_and_surface`, `ingest_and_flag_conflict`, `skip`) are handled with distinct behavior in the daemon.
- [ ] Malformed/unparseable LLM output falls back gracefully to `ingest_silent` rather than crashing the daemon.
- [ ] `/surfaced` shows reasoning-backed surfaced items and conflicts, not raw similarity hits.
- [ ] `tests/test_reasoner.py` passes with mocked LLM responses (no live API key needed for this test).
- [ ] Manually test: paste something that contradicts an existing memory (e.g., ingest "Passport renewed in January," then copy "Actually renewed my passport in June instead") and confirm it's flagged as `ingest_and_flag_conflict`, not silently stored as an unrelated duplicate.
