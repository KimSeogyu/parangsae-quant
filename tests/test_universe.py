from src.universe.model import compute_rankings, apply_hysteresis


def test_compute_rankings():
    volume_buffers = {
        "BTCUSDT": [1000.0, 2000.0, 3000.0],
        "ETHUSDT": [500.0, 600.0, 700.0],
        "XRPUSDT": [5000.0, 4000.0, 3000.0],
    }
    rankings = compute_rankings(volume_buffers)

    assert rankings["XRPUSDT"] == 1
    assert rankings["BTCUSDT"] == 2
    assert rankings["ETHUSDT"] == 3


def test_apply_hysteresis_inclusion():
    current_universe: set[str] = set()
    rankings = {"BTCUSDT": 1, "ETHUSDT": 50, "XRPUSDT": 101}

    new_universe, added, removed = apply_hysteresis(
        current_universe=current_universe,
        rankings=rankings,
        inclusion_rank=100,
        exclusion_rank=150,
    )

    assert new_universe == {"BTCUSDT", "ETHUSDT"}
    assert set(added) == {"BTCUSDT", "ETHUSDT"}
    assert removed == ()


def test_apply_hysteresis_retention():
    current_universe = {"BTCUSDT", "ETHUSDT"}
    rankings = {"BTCUSDT": 1, "ETHUSDT": 120}

    new_universe, added, removed = apply_hysteresis(
        current_universe=current_universe,
        rankings=rankings,
        inclusion_rank=100,
        exclusion_rank=150,
    )

    assert "ETHUSDT" in new_universe
    assert added == ()
    assert removed == ()


def test_apply_hysteresis_exclusion():
    current_universe = {"BTCUSDT", "ETHUSDT"}
    rankings = {"BTCUSDT": 1, "ETHUSDT": 200}

    new_universe, added, removed = apply_hysteresis(
        current_universe=current_universe,
        rankings=rankings,
        inclusion_rank=100,
        exclusion_rank=150,
    )

    assert "ETHUSDT" not in new_universe
    assert added == ()
    assert set(removed) == {"ETHUSDT"}


def test_apply_hysteresis_missing_from_rankings():
    """Symbols in universe but absent from rankings should be removed."""
    current_universe = {"BTCUSDT", "DELISTED"}
    rankings = {"BTCUSDT": 1}  # DELISTED not in rankings at all

    new_universe, added, removed = apply_hysteresis(
        current_universe=current_universe,
        rankings=rankings,
        inclusion_rank=100,
        exclusion_rank=150,
    )

    assert "DELISTED" not in new_universe
    assert set(removed) == {"DELISTED"}
