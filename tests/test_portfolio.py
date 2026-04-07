import pytest
from src.portfolio.construction import (
    combine_alphas,
    apply_constraints,
    compute_target_deltas,
)


def test_combine_alphas_weighted_sum():
    alpha_scores = {
        "momentum": {"BTC": 0.5, "ETH": -0.3},
        "mean_rev": {"BTC": -0.1, "ETH": 0.4},
    }
    weights = {"momentum": 0.6, "mean_rev": 0.4}
    universe = {"BTC", "ETH"}

    combined = combine_alphas(alpha_scores, weights, universe)

    assert combined["BTC"] == pytest.approx(0.26)
    assert combined["ETH"] == pytest.approx(-0.02)


def test_combine_alphas_ignores_non_universe():
    alpha_scores = {"momentum": {"BTC": 0.5, "XRP": 0.8}}
    weights = {"momentum": 1.0}
    universe = {"BTC"}

    combined = combine_alphas(alpha_scores, weights, universe)

    assert "BTC" in combined
    assert "XRP" not in combined


def test_apply_constraints_clips_position():
    raw_weights = {"BTC": 0.15, "ETH": 0.03, "SOL": -0.08}
    volatility = {"BTC": 0.02, "ETH": 0.03, "SOL": 0.05}

    result = apply_constraints(
        raw_weights=raw_weights,
        volatility=volatility,
        max_position_pct=0.05,
        max_total_exposure=1.0,
        risk_scale=1.0,
    )

    for w in result.values():
        assert abs(w) <= 0.05


def test_apply_constraints_risk_halt():
    raw_weights = {"BTC": 0.05}
    result = apply_constraints(
        raw_weights=raw_weights,
        volatility={"BTC": 0.02},
        max_position_pct=0.05,
        max_total_exposure=1.0,
        risk_scale=0.0,
    )
    assert result["BTC"] == 0.0


def test_compute_target_deltas():
    target_weights = {"BTC": 0.05, "ETH": 0.03}
    current_weights = {"BTC": 0.03, "ETH": 0.03, "SOL": 0.02}
    threshold = 0.005

    deltas = compute_target_deltas(target_weights, current_weights, threshold)

    assert deltas["BTC"] == pytest.approx(0.02)
    assert "ETH" not in deltas
    assert deltas["SOL"] == pytest.approx(-0.02)
