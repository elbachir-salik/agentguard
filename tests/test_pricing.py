import pytest

from agentguard.pricing import estimate_cost


def test_exact_model_match():
    cost = estimate_cost("gpt-4o", 1000, 1000)
    assert cost == pytest.approx(0.0125)


def test_dated_model_prefix_match():
    exact = estimate_cost("gpt-4o", 1000, 1000)
    dated = estimate_cost("gpt-4o-2024-08-06", 1000, 1000)
    assert dated == exact


def test_unknown_model_uses_fallback():
    cost = estimate_cost("unknown-model-xyz", 1000, 1000)
    assert cost == pytest.approx(0.004)
