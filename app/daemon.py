import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import pyperclip

from app.actions.registry import get_action
from app.agent.log_store import get_recent_decisions, log_decision
from app.agent.reasoner import decide
from app.agent.self_review import run_self_review
from app.config import settings
from app.embeddings import get_embedding_provider
from app.security.sensitive_filter import looks_sensitive
from app.storage import get_vector_store

SURFACED_PATH = Path("surfaced.json")


def _similarity(row: dict[str, Any]) -> float:
    if row.get("score_type") == "similarity" or "score" in row:
        return float(row.get("score", 0.0))
    return 1.0 - float(row.get("distance", 1.0))


def _append_surfaced(record: dict[str, Any]) -> None:
    existing = []
    if SURFACED_PATH.exists():
        existing = json.loads(SURFACED_PATH.read_text(encoding="utf-8"))
    existing.insert(0, record)
    SURFACED_PATH.write_text(json.dumps(existing[:100], indent=2), encoding="utf-8")


def read_surfaced() -> list[dict[str, Any]]:
    decisions = get_recent_decisions(n=100, notable_only=True)
    if decisions:
        return list(reversed(decisions))
    if not SURFACED_PATH.exists():
        return []
    return json.loads(SURFACED_PATH.read_text(encoding="utf-8"))


async def handle_new_clipboard_content(content: str) -> None:
    if len(content.strip()) < 20:
        return
    if looks_sensitive(content):
        return

    embedder = get_embedding_provider()
    store = get_vector_store()
    embedding = embedder.embed(content)

    near_dupes = store.query(embedding, 3)
    if near_dupes and _similarity(near_dupes[0]) > 0.95:
        return

    candidates = [row for row in store.query(embedding, 10) if _similarity(row) >= 0.15]
    decision = decide(content, candidates)
    log_decision(content, decision)

    if decision.action == "skip":
        return

    timestamp = time.time()
    memory_id = f"clipboard-{timestamp}"
    store.add(memory_id, embedding, content, {"source": "clipboard", "timestamp": timestamp})

    if decision.suggested_reminder:
        get_action("create_reminder").execute(text=decision.suggested_reminder)

    if decision.action == "ingest_and_surface":
        related = [row for row in candidates if row.get("id") in decision.related_memory_ids]
        _append_surfaced(
            {
                "clipboard_text": content,
                "ingested_ids": [memory_id],
                "related": related,
                "reasoning": decision.reasoning,
                "action": decision.action,
            }
        )
        print(f"[SURFACED] {content[:60]} relates to {[row.get('id') for row in related]}")
    elif decision.action == "ingest_and_flag_conflict":
        _append_surfaced(
            {
                "clipboard_text": content,
                "ingested_ids": [memory_id],
                "related_memory_ids": decision.related_memory_ids,
                "reasoning": decision.reasoning,
                "conflict_summary": decision.conflict_summary,
                "action": decision.action,
            }
        )
        print(f"[CONFLICT] {content[:60]} - {decision.conflict_summary}")


async def watch_clipboard(poll_interval: float = 1.0) -> None:
    last_hash = ""
    while True:
        try:
            content = pyperclip.paste()
            if content:
                digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
                if digest != last_hash:
                    last_hash = digest
                    await handle_new_clipboard_content(content)
        except Exception as exc:
            print(f"clipboard daemon error: {exc}")
        await asyncio.sleep(poll_interval)


async def self_review_loop() -> None:
    while True:
        try:
            flagged = run_self_review()
            print(f"[SELF-REVIEW] Completed. {len(flagged)} items flagged.")
        except Exception as exc:
            print(f"[SELF-REVIEW] Error: {exc}")
        await asyncio.sleep(settings.self_review_interval_hours * 3600)


async def run_daemon() -> None:
    await asyncio.gather(watch_clipboard(), self_review_loop())


def main() -> None:
    asyncio.run(run_daemon())


if __name__ == "__main__":
    main()
