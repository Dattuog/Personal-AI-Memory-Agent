from app.config import settings
from app.storage.base import VectorStore


def get_vector_store() -> VectorStore:
    if settings.vector_store == "chroma":
        from app.storage.chroma_store import ChromaStore

        return ChromaStore()
    if settings.vector_store == "pinecone":
        from app.storage.pinecone_store import PineconeStore

        return PineconeStore()
    raise ValueError(f"Unsupported VECTOR_STORE: {settings.vector_store}")
