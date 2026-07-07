from typing import Any

import chromadb

from app.config import settings
from app.storage.base import VectorStore


class ChromaStore(VectorStore):
    def __init__(self) -> None:
        if settings.chroma_host:
            self.client = chromadb.HttpClient(host=settings.chroma_host)
        else:
            self.client = chromadb.PersistentClient(path=settings.chroma_path)
        self.collection = self.client.get_or_create_collection(
            name="memories",
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, id: str, embedding: list[float], text: str, metadata: dict[str, Any]) -> None:
        clean_metadata = {
            key: value if isinstance(value, str | int | float | bool) or value is None else str(value)
            for key, value in metadata.items()
        }
        self.collection.upsert(
            ids=[id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[clean_metadata],
        )

    def query(self, embedding: list[float], k: int) -> list[dict[str, Any]]:
        result = self.collection.query(query_embeddings=[embedding], n_results=k)
        rows: list[dict[str, Any]] = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        for item_id, text, metadata, distance in zip(ids, docs, metadatas, distances, strict=False):
            rows.append(
                {
                    "id": item_id,
                    "text": text,
                    "metadata": metadata or {},
                    "distance": float(distance),
                    "score_type": "distance",
                }
            )
        return rows

    def delete(self, id: str) -> None:
        self.collection.delete(ids=[id])
