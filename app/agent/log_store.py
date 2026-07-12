import json
import time
from pathlib import Path
from typing import Any

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
        "suggested_reminder": decision.suggested_reminder,
    }
    existing: list[dict[str, Any]] = []
    if LOG_PATH.exists():
        existing = json.loads(LOG_PATH.read_text(encoding="utf-8"))
    existing.append(entry)
    LOG_PATH.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def get_recent_decisions(n: int = 20, notable_only: bool = False) -> list[dict[str, Any]]:
    if not LOG_PATH.exists():
        return []
    existing = json.loads(LOG_PATH.read_text(encoding="utf-8"))
    if notable_only:
        existing = [item for item in existing if item.get("action") != "ingest_silent"]
    return existing[-n:]
