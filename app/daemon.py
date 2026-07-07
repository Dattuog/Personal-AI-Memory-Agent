import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import pyperclip

from app.config import settings
from app.embeddings import get_embedding_provider
from app.ingestion.ingest import ingest_entry
from app.security.sensitive_filter import looks_sensitive
from app.storage import get_vector_store

SURFACED_PATH = Path("surfaced.json")


def _similarity(row: dict[str, Any]) -> float:
    if row.get("score_type") == "similarity" or "score" in row:
        return float(row.get("score", 0.0))
    return 1.0 - float(row.get("distance", 1.0))


def _append_surfaced(record: dict[str, Any]) -> None:
    existing = read_surfaced()
    existing.insert(0, record)
    SURFACED_PATH.write_text(json.dumps(existing[:100], indent=2), encoding="utf-8")


def read_surfaced() -> list[dict[str, Any]]:
    if not SURFACED_PATH.exists():
        return []
    return json.loads(SURFACED_PATH.read_text(encoding="utf-8"))


async def watch_clipboard() -> None:
    last_hash = ""
    embedder = get_embedding_provider()
    store = get_vector_store()
    while True:
        try:
            content = pyperclip.paste()
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
            if digest != last_hash:
                last_hash = digest
                if len(content.strip()) >= 20 and not looks_sensitive(content):
                    embedding = embedder.embed(content)
                    duplicates = store.query(embedding, 1)
                    duplicate = duplicates and _similarity(duplicates[0]) > 0.95
                    if not duplicate:
                        ids = ingest_entry(content, source="clipboard", tags=[])
                        related = [
                            row
                            for row in store.query(embedding, 10)
                            if row.get("id") not in ids
                            and _similarity(row) > settings.similarity_threshold
                        ]
                        if related:
                            _append_surfaced({"clipboard_text": content, "ingested_ids": ids, "related": related})
        except Exception as exc:
            print(f"clipboard daemon error: {exc}")
        await asyncio.sleep(1)


def main() -> None:
    asyncio.run(watch_clipboard())


if __name__ == "__main__":
    main()
