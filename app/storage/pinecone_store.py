from typing import Any

from pinecone import Pinecone

from app.config import settings
from app.storage.base import VectorStore


class PineconeStore(VectorStore):
    def __init__(self) -> None:
        if not settings.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY is required when VECTOR_STORE=pinecone")
        self.client = Pinecone(api_key=settings.pinecone_api_key)
        self.index = self.client.Index(settings.pinecone_index)

    def add(self, id: str, embedding: list[float], text: str, metadata: dict[str, Any]) -> None:
        payload = dict(metadata)
        payload["text"] = text
        self.index.upsert(vectors=[{"id": id, "values": embedding, "metadata": payload}])

    def query(self, embedding: list[float], k: int) -> list[dict[str, Any]]:
        result = self.index.query(vector=embedding, top_k=k, include_metadata=True)
        rows: list[dict[str, Any]] = []
        for match in result.get("matches", []):
            metadata = match.get("metadata") or {}
            rows.append(
                {
                    "id": match["id"],
                    "text": metadata.get("text", ""),
                    "metadata": {key: value for key, value in metadata.items() if key != "text"},
                    "score": float(match.get("score", 0.0)),
                    "score_type": "similarity",
                }
            )
        return rows

    def delete(self, id: str) -> None:
        self.index.delete(ids=[id])

    def list_all(self, limit: int = 1000) -> list[dict[str, Any]]:
        raise NotImplementedError("Scheduled self-review list_all is currently implemented for Chroma only.")

    def update_metadata(self, id: str, metadata: dict[str, Any]) -> None:
        self.index.update(id=id, set_metadata=metadata)
