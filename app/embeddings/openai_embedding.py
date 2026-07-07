from openai import OpenAI

from app.config import settings
from app.embeddings.base import EmbeddingProvider


class OpenAIEmbedding(EmbeddingProvider):
    def __init__(self) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai")
        self.client = OpenAI(api_key=settings.openai_api_key)

    def embed(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model="text-embedding-3-small", input=text)
        return response.data[0].embedding
