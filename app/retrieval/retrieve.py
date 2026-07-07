import time
from typing import Any

from app.config import settings
from app.embeddings import get_embedding_provider
from app.retrieval.rerank import blended_score, decay_score
from app.storage import get_vector_store


def _cosine_similarity(candidate: dict[str, Any]) -> float:
    if candidate.get("score_type") == "similarity" or "score" in candidate:
        return float(candidate.get("score", 0.0))
    return 1.0 - float(candidate.get("distance", 1.0))


def retrieve_and_rerank(query: str, top_k: int) -> list[dict[str, Any]]:
    embedding = get_embedding_provider().embed(query)
    candidates = get_vector_store().query(embedding, max(20, top_k * 4))
    now = time.time()
    ranked: list[dict[str, Any]] = []
    for candidate in candidates:
        cosine = _cosine_similarity(candidate)
        if cosine < settings.similarity_threshold:
            continue
        metadata = candidate.get("metadata") or {}
        timestamp = float(metadata.get("timestamp", now))
        decay = decay_score(timestamp, now, settings.decay_half_life_days)
        score = blended_score(cosine, decay, settings.rerank_alpha)
        ranked.append(
            {
                "id": candidate.get("id"),
                "text": candidate.get("text", ""),
                "metadata": metadata,
                "cosine_similarity": cosine,
                "decay_score": decay,
                "score": score,
            }
        )
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:top_k]
