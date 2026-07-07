from sentence_transformers import SentenceTransformer

from app.embeddings.base import EmbeddingProvider


class LocalEmbedding(EmbeddingProvider):
    def __init__(self) -> None:
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(self, text: str) -> list[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()
