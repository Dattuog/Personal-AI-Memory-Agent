from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    vector_store: str = "chroma"
    embedding_provider: str = "local"
    llm_provider: str = "groq"
    chroma_path: str = "./chroma_data"
    chroma_host: str = ""
    pinecone_api_key: str = ""
    pinecone_index: str = "memories"
    openai_api_key: str = ""
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    rerank_alpha: float = 0.7
    decay_half_life_days: float = 7.0
    similarity_threshold: float = 0.75
    self_review_interval_hours: float = 24.0
    self_review_cooldown_hours: float = 48.0


settings = Settings()
