"""Tests for research pipeline configuration."""

from src.research.config import (
    LabelConfig,
    ModelSettings,
    ResearchPortfolioConfig,
    load_research_settings,
)


class TestLabelConfig:
    def test_weights_sum_to_one(self):
        cfg = LabelConfig(primary_weight=0.7, secondary_weight=0.3)
        assert abs(cfg.primary_weight + cfg.secondary_weight - 1.0) < 1e-6

    def test_weights_invalid_raises(self):
        import pytest

        with pytest.raises(ValueError, match="must sum to 1.0"):
            LabelConfig(primary_weight=0.5, secondary_weight=0.3)


class TestPortfolioConfig:
    def test_valid_direction(self):
        cfg = ResearchPortfolioConfig(direction="long_short")
        assert cfg.direction == "long_short"

    def test_invalid_direction_raises(self):
        import pytest

        with pytest.raises(ValueError, match="direction must be"):
            ResearchPortfolioConfig(direction="market_neutral")


class TestLoadSettings:
    def test_load_from_yaml(self):
        settings = load_research_settings()
        assert settings.universe.liquidity.top_n == 30
        assert settings.universe.inclusion.min_listing_days == 180
        assert settings.model.labels.primary_horizon_minutes == 20
        assert settings.model.labels.secondary_horizon_minutes == 60
        assert settings.model.training.train_window_days == 90
        assert settings.model.ridge.alpha_grid[0] == 0.001
        assert settings.model.portfolio.long_count == 6
        assert settings.model.portfolio.short_count == 6
        assert settings.model.portfolio.single_name_cap == 0.12
        assert settings.model.portfolio.btc_beta_cap == 0.10

    def test_universe_exclusion_categories(self):
        settings = load_research_settings()
        cats = settings.universe.exclusion.categories
        assert "stablecoin" in cats
        assert "tokenized_gold" in cats
        assert "leveraged_token" in cats

    def test_model_settings_defaults(self):
        ms = ModelSettings()
        assert ms.data.research_bar_minutes == 5
        assert ms.data.decision_bar_minutes == 20
        assert ms.preprocessing.clip == 5.0
        assert ms.execution.max_requotes == 2
        assert ms.postfill.reduce_pct == 0.50

    def test_sector_map_loaded(self):
        settings = load_research_settings()
        sector_map = settings.universe.sector_map
        assert "l1" in sector_map
        assert "BTC" in sector_map["l1"]
        assert "defi" in sector_map
