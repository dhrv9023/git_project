"""
tests/unit/test_config.py — Unit tests for AppConfig env overrides.
"""
import os
import pytest
from core.config import AppConfig


class TestAppConfigDefaults:
    def test_default_ticker(self):
        cfg = AppConfig()
        assert cfg.default_ticker == "AAPL"

    def test_default_environment(self):
        cfg = AppConfig()
        assert cfg.environment == "development"

    def test_risk_free_rate_positive(self):
        cfg = AppConfig()
        assert 0 < cfg.risk_free_rate_annual < 1

    def test_transaction_cost_positive(self):
        cfg = AppConfig()
        assert cfg.transaction_cost_pct > 0


class TestAppConfigEnvOverrides:
    def test_int_override(self, monkeypatch):
        monkeypatch.setenv("SB_EPOCHS", "99")
        cfg = AppConfig.from_env()
        assert cfg.epochs == 99

    def test_float_override(self, monkeypatch):
        monkeypatch.setenv("SB_LEARNING_RATE", "0.001")
        cfg = AppConfig.from_env()
        assert abs(cfg.learning_rate - 0.001) < 1e-10

    def test_bool_override_true(self, monkeypatch):
        monkeypatch.setenv("SB_DEBUG", "true")
        cfg = AppConfig.from_env()
        assert cfg.debug is True

    def test_bool_override_false(self, monkeypatch):
        monkeypatch.setenv("SB_DEBUG", "false")
        cfg = AppConfig.from_env()
        assert cfg.debug is False

    def test_bool_override_1(self, monkeypatch):
        monkeypatch.setenv("SB_DEBUG", "1")
        cfg = AppConfig.from_env()
        assert cfg.debug is True

    def test_str_override(self, monkeypatch):
        monkeypatch.setenv("SB_DEFAULT_TICKER", "MSFT")
        cfg = AppConfig.from_env()
        assert cfg.default_ticker == "MSFT"

    def test_port_from_standard_env(self, monkeypatch):
        monkeypatch.setenv("PORT", "8080")
        cfg = AppConfig.from_env()
        assert cfg.port == 8080

    def test_environment_from_standard_env(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "production")
        cfg = AppConfig.from_env()
        assert cfg.environment == "production"

    def test_invalid_int_keeps_default(self, monkeypatch):
        monkeypatch.setenv("SB_EPOCHS", "notanumber")
        cfg = AppConfig.from_env()
        assert cfg.epochs == AppConfig().epochs


class TestAppConfigPaths:
    def test_registry_path(self):
        cfg = AppConfig(model_artifacts_dir="artifacts", registry_filename="reg.json")
        assert cfg.registry_path == "artifacts/reg.json"

    def test_cache_dir(self):
        cfg = AppConfig(model_artifacts_dir="artifacts", inference_cache_dir="cache")
        assert cfg.cache_dir == "artifacts/cache"
