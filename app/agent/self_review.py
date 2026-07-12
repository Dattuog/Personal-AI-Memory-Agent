import json
import time
from datetime import datetime
from typing import Any

from app.actions.registry import get_action
from app.agent.log_store import log_decision
from app.agent.schema import AgentDecision
from app.config import settings
from app.storage import get_vector_store
from app.synthesis import get_llm_provider

SELF_REVIEW_SYSTEM_PROMPT = """You are the self-review layer of a personal memory agent. You are given a batch of stored memories along with today's date. Review them for anything time-sensitive: upcoming deadlines, unresolved tasks, mentioned dates that are approaching, or things the user said they needed to do but likely has not confirmed doing.

Respond with ONLY a JSON array, no other text, where each element has this shape:
{
  "memory_id": "<id>",
  "needs_attention": true | false,
  "reminder_text": "<short reminder text, or null if needs_attention is false>",
  "reasoning": "<one sentence>"
}

Only mark needs_attention as true for genuinely time-sensitive items.
"""


def _build_batch_message(memories: list[dict[str, Any]]) -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    lines = [f"Today's date: {today}", "", "Memories:"]
    for memory in memories:
        metadata = memory.get("metadata") or {}
        ts = metadata.get("timestamp")
        date_str = datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d") if ts else "unknown"
        lines.append(f"id: {memory.get('id')} | stored: {date_str}\ntext: {memory.get('text', '')}")
    return "\n\n".join(lines)


def run_self_review(batch_size: int = 25) -> list[dict[str, Any]]:
    store = get_vector_store()
    llm = get_llm_provider()
    all_memories = store.list_all(limit=1000)

    now = time.time()
    cooldown_seconds = settings.self_review_cooldown_hours * 3600
    candidates = [
        memory
        for memory in all_memories
        if now - float((memory.get("metadata") or {}).get("last_reviewed_at", 0)) > cooldown_seconds
    ]

    results: list[dict[str, Any]] = []
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]
        if not batch:
            continue

        raw = llm.complete(
            system_prompt=SELF_REVIEW_SYSTEM_PROMPT,
            user_message=_build_batch_message(batch),
            temperature=0.1,
        )
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = []

        for item in parsed if isinstance(parsed, list) else []:
            memory_id = item.get("memory_id")
            if not memory_id:
                continue

            store.update_metadata(memory_id, {"last_reviewed_at": now})

            if item.get("needs_attention") and item.get("reminder_text"):
                reminder_text = item["reminder_text"]
                get_action("create_reminder").execute(text=reminder_text)
                decision = AgentDecision(
                    action="self_review_flag",
                    reasoning=item.get("reasoning", ""),
                    related_memory_ids=[memory_id],
                    suggested_reminder=reminder_text,
                )
                log_decision(reminder_text, decision)
                results.append({"memory_id": memory_id, "reminder_text": reminder_text})

    return results
