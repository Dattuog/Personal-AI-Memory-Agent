def decay_score(entry_timestamp: float, now: float, half_life_days: float) -> float:
    age_days = max(0.0, (now - entry_timestamp) / 86400)
    return 0.5 ** (age_days / half_life_days)


def blended_score(cosine_sim: float, decay: float, alpha: float) -> float:
    return alpha * cosine_sim + (1 - alpha) * decay
