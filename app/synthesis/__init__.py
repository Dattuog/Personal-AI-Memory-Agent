from app.config import settings
from app.synthesis.base import LLMProvider


def get_llm_provider() -> LLMProvider:
    if settings.llm_provider == "groq":
        from app.synthesis.groq_client import GroqProvider

        return GroqProvider(api_key=settings.groq_api_key, model=settings.groq_model)
    raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")
