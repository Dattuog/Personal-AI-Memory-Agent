from app.config import settings
from app.embeddings.base import EmbeddingProvider


def get_embedding_provider() -> EmbeddingProvider:
    if settings.embedding_provider == "local":
        from app.embeddings.local_embedding import LocalEmbedding

        return LocalEmbedding()
    if settings.embedding_provider == "openai":
        from app.embeddings.openai_embedding import OpenAIEmbedding

        return OpenAIEmbedding()
    raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {settings.embedding_provider}")
