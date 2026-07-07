import pytest

from app.retrieval.rerank import blended_score, decay_score


def test_decay_score_boundaries_and_monotonicity() -> None:
    now = 1_000_000.0
    half_life_days = 7.0
    assert decay_score(now, now, half_life_days) == pytest.approx(1.0)
    assert decay_score(now - half_life_days * 86400, now, half_life_days) == pytest.approx(0.5)
    assert decay_score(now - 2 * half_life_days * 86400, now, half_life_days) < 0.5


def test_blended_score_alpha_edges() -> None:
    assert blended_score(0.2, 0.9, 1.0) == pytest.approx(0.2)
    assert blended_score(0.2, 0.9, 0.0) == pytest.approx(0.9)
    assert blended_score(0.8, 0.6, 0.7) == pytest.approx(0.74)
