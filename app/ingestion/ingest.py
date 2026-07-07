import time
import uuid

from app.embeddings import get_embedding_provider
from app.ingestion.chunker import chunk_text
from app.storage import get_vector_store


def ingest_entry(text: str, source: str, tags: list[str]) -> list[str]:
    chunks = chunk_text(text)
    if not chunks:
        return []
    embedder = get_embedding_provider()
    store = get_vector_store()
    parent_id = str(uuid.uuid4())
    timestamp = time.time()
    ids: list[str] = []
    for index, chunk in enumerate(chunks):
        chunk_id = f"{parent_id}:{index}"
        metadata = {
            "source": source,
            "tags": ",".join(tags),
            "timestamp": timestamp,
            "parent_id": parent_id,
            "chunk_index": index,
        }
        store.add(chunk_id, embedder.embed(chunk), chunk, metadata)
        ids.append(chunk_id)
    return ids
