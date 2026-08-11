"""
tests/unit/test_p1_features.py — Unit tests for P1 sprint deliverables.

Covers:
  - AuthService: registration, authentication, roles, duplicate guard
  - WorkspaceStore: CRUD, watchlists, alerts, thread safety, atomicity
  - ComparisonService: normalisation, correlation, summary stats, error handling

Engineering decision: All I/O is redirected to tmp directories (tmp_path
pytest fixture). No real yfinance calls — ComparisonService is tested via
direct injection of synthetic DataFrames into _normalise_prices,
_correlation_matrix, and _summary_stats private helpers, which are pure
functions of the data they receive.
"""
from __future__ import annotations

import os
import threading
import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_synthetic_ohlcv


# ═══════════════════════════════════════════════════════════════════════════════
# AuthService
# ═══════════════════════════════════════════════════════════════════════════════

class TestAuthServiceRegistration:
    """Tests for user creation and validation."""

    @pytest.fixture
    def svc(self, tmp_path):
        from app.auth.auth_service import AuthService
        return AuthService(users_file=str(tmp_path / "users.json"))

    def test_register_returns_public_dict_without_hash(self, svc):
        user = svc.register("alice", "securepass1", "analyst")
        assert "password_hash" not in user
        assert user["username"] == "alice"

    def test_register_stores_role_correctly(self, svc):
        user = svc.register("bob", "securepass1", "quant")
        assert user["role"] == "quant"

    def test_register_username_lowercased(self, svc):
        user = svc.register("Charlie", "securepass1", "analyst")
        assert user["username"] == "charlie"

    def test_register_duplicate_raises_value_error(self, svc):
        svc.register("dave", "securepass1", "analyst")
        with pytest.raises(ValueError, match="already exists"):
            svc.register("Dave", "anotherpass", "analyst")

    def test_register_short_username_raises(self, svc):
        with pytest.raises(ValueError, match="3 characters"):
            svc.register("ab", "securepass1", "analyst")

    def test_register_short_password_raises(self, svc):
        with pytest.raises(ValueError, match="8 characters"):
            svc.register("validuser", "short", "analyst")

    def test_register_analyst_permissions(self, svc):
        user = svc.register("analyst1", "securepass1", "analyst")
        perms = user["permissions"]
        assert "read" in perms
        assert "write" in perms
        assert "execute_models" in perms
        assert "manage_users" not in perms

    def test_register_admin_has_all_permissions(self, svc):
        user = svc.register("adminuser", "securepass1", "admin")
        perms = user["permissions"]
        assert "manage_users" in perms
        assert "flush_cache" in perms
        assert "portfolio_opt" in perms

    def test_user_persisted_across_instances(self, tmp_path):
        from app.auth.auth_service import AuthService
        f = str(tmp_path / "users.json")
        AuthService(users_file=f).register("persist_user", "securepass1", "analyst")
        # New instance, same file
        svc2 = AuthService(users_file=f)
        assert svc2.user_exists("persist_user")


class TestAuthServiceAuthentication:
    """Tests for credential verification."""

    @pytest.fixture
    def svc(self, tmp_path):
        from app.auth.auth_service import AuthService
        s = AuthService(users_file=str(tmp_path / "users.json"))
        s.register("testuser", "correctpassword", "analyst")
        return s

    def test_authenticate_correct_credentials(self, svc):
        user = svc.authenticate("testuser", "correctpassword")
        assert user is not None
        assert user["username"] == "testuser"

    def test_authenticate_wrong_password_returns_none(self, svc):
        assert svc.authenticate("testuser", "wrongpassword") is None

    def test_authenticate_unknown_user_returns_none(self, svc):
        assert svc.authenticate("ghost", "anypassword") is None

    def test_authenticate_case_insensitive_username(self, svc):
        user = svc.authenticate("TESTUSER", "correctpassword")
        assert user is not None

    def test_authenticate_no_password_hash_in_result(self, svc):
        user = svc.authenticate("testuser", "correctpassword")
        assert "password_hash" not in user

    def test_get_user_returns_public_profile(self, svc):
        user = svc.get_user("testuser")
        assert user is not None
        assert user["username"] == "testuser"
        assert "password_hash" not in user

    def test_get_user_missing_returns_none(self, svc):
        assert svc.get_user("nobody") is None

    def test_user_exists_true(self, svc):
        assert svc.user_exists("testuser") is True

    def test_user_exists_false(self, svc):
        assert svc.user_exists("ghost") is False


# ═══════════════════════════════════════════════════════════════════════════════
# WorkspaceStore
# ═══════════════════════════════════════════════════════════════════════════════

class TestWorkspaceStoreCRUD:
    """Tests for workspace save / get / list / delete."""

    @pytest.fixture
    def store(self, tmp_path):
        from storage.workspace_store import WorkspaceStore
        return WorkspaceStore(base_dir=str(tmp_path / "ws"))

    def test_save_and_get_workspace(self, store):
        cfg = {"tickers": ["AAPL"], "theme": "dark"}
        store.save_workspace("user1", "my_view", cfg)
        ws = store.get_workspace("user1", "my_view")
        assert ws is not None
        assert ws["config"]["tickers"] == ["AAPL"]
        assert ws["name"] == "my_view"
        assert ws["user_id"] == "user1"

    def test_get_nonexistent_workspace_returns_none(self, store):
        assert store.get_workspace("user1", "nope") is None

    def test_list_workspaces_returns_only_user_workspaces(self, store):
        store.save_workspace("user1", "ws_a", {})
        store.save_workspace("user1", "ws_b", {})
        store.save_workspace("user2", "ws_c", {})  # different user
        result = store.list_workspaces("user1")
        assert len(result) == 2
        names = {w["name"] for w in result}
        assert names == {"ws_a", "ws_b"}

    def test_list_workspaces_empty_for_unknown_user(self, store):
        assert store.list_workspaces("ghost") == []

    def test_save_overwrites_existing(self, store):
        store.save_workspace("user1", "ws", {"v": 1})
        store.save_workspace("user1", "ws", {"v": 2})
        ws = store.get_workspace("user1", "ws")
        assert ws["config"]["v"] == 2

    def test_delete_workspace_returns_true(self, store):
        store.save_workspace("user1", "todelete", {})
        deleted = store.delete_workspace("user1", "todelete")
        assert deleted is True
        assert store.get_workspace("user1", "todelete") is None

    def test_delete_nonexistent_returns_false(self, store):
        assert store.delete_workspace("user1", "ghost") is False

    def test_workspace_has_updated_at_timestamp(self, store):
        store.save_workspace("user1", "ts_ws", {})
        ws = store.get_workspace("user1", "ts_ws")
        assert "updated_at" in ws
        assert ws["updated_at"] is not None


class TestWorkspaceStoreWatchlist:
    """Tests for watchlist add / get / remove."""

    @pytest.fixture
    def store(self, tmp_path):
        from storage.workspace_store import WorkspaceStore
        return WorkspaceStore(base_dir=str(tmp_path / "ws"))

    def test_empty_watchlist_for_new_user(self, store):
        assert store.get_watchlist("u1") == []

    def test_add_ticker_uppercased(self, store):
        store.add_to_watchlist("u1", "aapl")
        assert "AAPL" in store.get_watchlist("u1")

    def test_add_ticker_no_duplicates(self, store):
        store.add_to_watchlist("u1", "AAPL")
        store.add_to_watchlist("u1", "AAPL")
        assert store.get_watchlist("u1").count("AAPL") == 1

    def test_add_multiple_tickers(self, store):
        store.add_to_watchlist("u1", "AAPL")
        store.add_to_watchlist("u1", "MSFT")
        wl = store.get_watchlist("u1")
        assert "AAPL" in wl and "MSFT" in wl

    def test_remove_ticker(self, store):
        store.add_to_watchlist("u1", "AAPL")
        store.remove_from_watchlist("u1", "AAPL")
        assert "AAPL" not in store.get_watchlist("u1")

    def test_remove_nonexistent_ticker_noop(self, store):
        store.add_to_watchlist("u1", "AAPL")
        store.remove_from_watchlist("u1", "TSLA")  # was never added
        assert store.get_watchlist("u1") == ["AAPL"]

    def test_watchlists_are_user_isolated(self, store):
        store.add_to_watchlist("u1", "AAPL")
        store.add_to_watchlist("u2", "MSFT")
        assert "MSFT" not in store.get_watchlist("u1")
        assert "AAPL" not in store.get_watchlist("u2")


class TestWorkspaceStoreAlerts:
    """Tests for alert creation, retrieval, and status updates."""

    @pytest.fixture
    def store(self, tmp_path):
        from storage.workspace_store import WorkspaceStore
        return WorkspaceStore(base_dir=str(tmp_path / "ws"))

    def test_create_alert_returns_dict_with_id(self, store):
        alert = store.create_alert("u1", "AAPL", "PRICE_ABOVE", 200.0)
        assert "alert_id" in alert
        assert alert["alert_id"].startswith("alt_")

    def test_created_alert_is_active(self, store):
        alert = store.create_alert("u1", "AAPL", "PRICE_ABOVE", 200.0)
        assert alert["status"] == "ACTIVE"

    def test_created_alert_ticker_uppercased(self, store):
        alert = store.create_alert("u1", "aapl", "PRICE_ABOVE", 200.0)
        assert alert["ticker"] == "AAPL"

    def test_get_alerts_returns_user_alerts_only(self, store):
        store.create_alert("u1", "AAPL", "PRICE_ABOVE", 200.0)
        store.create_alert("u2", "MSFT", "PRICE_BELOW", 100.0)
        alerts = store.get_alerts("u1")
        assert all(a["user_id"] == "u1" for a in alerts)

    def test_get_alerts_filter_by_status(self, store):
        alert = store.create_alert("u1", "AAPL", "PRICE_ABOVE", 200.0)
        store.update_alert_status(alert["alert_id"], "TRIGGERED")
        active = store.get_alerts("u1", status="ACTIVE")
        triggered = store.get_alerts("u1", status="TRIGGERED")
        assert len(active) == 0
        assert len(triggered) == 1

    def test_update_alert_status(self, store):
        alert = store.create_alert("u1", "AAPL", "PRICE_ABOVE", 200.0)
        updated = store.update_alert_status(alert["alert_id"], "TRIGGERED", "2024-01-01T00:00:00")
        assert updated["status"] == "TRIGGERED"
        assert updated["triggered_at"] == "2024-01-01T00:00:00"

    def test_update_nonexistent_alert_returns_none(self, store):
        assert store.update_alert_status("fake_id", "TRIGGERED") is None

    def test_delete_alert(self, store):
        alert = store.create_alert("u1", "AAPL", "PRICE_ABOVE", 200.0)
        deleted = store.delete_alert("u1", alert["alert_id"])
        assert deleted is True
        assert store.get_alerts("u1") == []

    def test_delete_alert_wrong_user_returns_false(self, store):
        alert = store.create_alert("u1", "AAPL", "PRICE_ABOVE", 200.0)
        assert store.delete_alert("u2", alert["alert_id"]) is False

    def test_get_all_active_alerts(self, tmp_path):
        """Isolated store so no state bleeds in from other tests."""
        import time as _time
        from storage.workspace_store import WorkspaceStore
        isolated = WorkspaceStore(base_dir=str(tmp_path / "isolated"))
        a1 = isolated.create_alert("u1", "AAPL", "PRICE_ABOVE", 200.0)
        _time.sleep(0.002)  # ensure unique ms-based alert_id
        a2 = isolated.create_alert("u2", "MSFT", "PRICE_BELOW", 100.0)
        isolated.update_alert_status(a2["alert_id"], "TRIGGERED")
        active = isolated.get_all_active_alerts()
        assert len(active) == 1
        assert active[0]["ticker"] == "AAPL"


class TestWorkspaceStoreThreadSafety:
    """Concurrent writes must not corrupt the store."""

    def test_concurrent_watchlist_writes_no_corruption(self, tmp_path):
        from storage.workspace_store import WorkspaceStore
        store = WorkspaceStore(base_dir=str(tmp_path / "ws"))
        tickers = [f"T{i:03d}" for i in range(20)]
        errors = []

        def add_ticker(t):
            try:
                store.add_to_watchlist("user1", t)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_ticker, args=(t,)) for t in tickers]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert errors == [], f"Concurrent write errors: {errors}"
        wl = store.get_watchlist("user1")
        # All unique
        assert len(wl) == len(set(wl))


# ═══════════════════════════════════════════════════════════════════════════════
# ComparisonService (pure-function layer — no yfinance)
# ═══════════════════════════════════════════════════════════════════════════════

def _make_aligned(tickers: list[str], n: int = 200) -> dict[str, pd.DataFrame]:
    """Create aligned synthetic engineered DataFrames for each ticker."""
    from ml.features import engineer_features
    aligned = {}
    for i, t in enumerate(tickers):
        raw = make_synthetic_ohlcv(n=n + 60, seed=i * 7)   # extra rows for dropna
        df = engineer_features(raw)
        aligned[t] = df.iloc[:n]
    return aligned


class TestComparisonServiceNormalisation:
    @pytest.fixture
    def svc(self, test_cfg):
        from app.services.comparison_service import ComparisonService
        from app.repositories.market_data_repo import MarketDataRepository
        repo = MarketDataRepository()
        return ComparisonService(repo, test_cfg)

    def test_normalised_prices_start_at_100(self, svc):
        aligned = _make_aligned(["AAPL", "MSFT"])
        result = svc._normalise_prices(aligned)
        for ticker, series in result.items():
            assert series[0]["value"] == pytest.approx(100.0), f"{ticker} does not start at 100"

    def test_normalised_prices_returns_correct_keys(self, svc):
        aligned = _make_aligned(["AAPL"])
        result = svc._normalise_prices(aligned)
        assert "AAPL" in result
        assert "date" in result["AAPL"][0]
        assert "value" in result["AAPL"][0]

    def test_normalised_length_matches_input(self, svc):
        aligned = _make_aligned(["AAPL"], n=150)
        result = svc._normalise_prices(aligned)
        assert len(result["AAPL"]) == len(aligned["AAPL"])


class TestComparisonServiceCorrelation:
    @pytest.fixture
    def svc(self, test_cfg):
        from app.services.comparison_service import ComparisonService
        from app.repositories.market_data_repo import MarketDataRepository
        repo = MarketDataRepository()
        return ComparisonService(repo, test_cfg)

    def test_correlation_matrix_diagonal_is_one(self, svc):
        aligned = _make_aligned(["AAPL", "MSFT", "GOOG"])
        result = svc._correlation_matrix(aligned)
        matrix = result["matrix"]
        for i in range(len(matrix)):
            assert matrix[i][i] == pytest.approx(1.0, abs=1e-6)

    def test_correlation_matrix_is_symmetric(self, svc):
        aligned = _make_aligned(["AAPL", "MSFT"])
        result = svc._correlation_matrix(aligned)
        m = result["matrix"]
        assert m[0][1] == pytest.approx(m[1][0], abs=1e-6)

    def test_correlation_labels_contain_all_tickers(self, svc):
        tickers = ["AAPL", "MSFT", "TSLA"]
        aligned = _make_aligned(tickers)
        result = svc._correlation_matrix(aligned)
        for t in tickers:
            assert t in result["labels"]

    def test_correlation_values_bounded_minus1_to_1(self, svc):
        aligned = _make_aligned(["AAPL", "MSFT"])
        result = svc._correlation_matrix(aligned)
        for row in result["matrix"]:
            for v in row:
                assert -1.0 <= v <= 1.0, f"Correlation out of bounds: {v}"


class TestComparisonServiceSummaryStats:
    @pytest.fixture
    def svc(self, test_cfg):
        from app.services.comparison_service import ComparisonService
        from app.repositories.market_data_repo import MarketDataRepository
        repo = MarketDataRepository()
        return ComparisonService(repo, test_cfg)

    def test_summary_has_required_keys(self, svc):
        aligned = _make_aligned(["AAPL"])
        result = svc._summary_stats(aligned)
        expected_keys = {
            "total_return_pct", "annualised_return_pct",
            "annualised_volatility_pct", "sharpe_ratio",
            "max_drawdown_pct", "latest_rsi", "latest_close", "start_close",
        }
        assert expected_keys.issubset(result["AAPL"].keys())

    def test_max_drawdown_is_nonpositive(self, svc):
        """Max drawdown is always ≤ 0 by definition."""
        aligned = _make_aligned(["AAPL"])
        result = svc._summary_stats(aligned)
        assert result["AAPL"]["max_drawdown_pct"] <= 0.0

    def test_volatility_is_positive(self, svc):
        aligned = _make_aligned(["AAPL"])
        result = svc._summary_stats(aligned)
        assert result["AAPL"]["annualised_volatility_pct"] > 0.0

    def test_rsi_is_within_0_100(self, svc):
        aligned = _make_aligned(["AAPL"])
        result = svc._summary_stats(aligned)
        rsi = result["AAPL"]["latest_rsi"]
        if rsi is not None:
            assert 0.0 <= rsi <= 100.0

    def test_summary_computed_for_all_tickers(self, svc):
        tickers = ["AAPL", "MSFT", "GOOG"]
        aligned = _make_aligned(tickers)
        result = svc._summary_stats(aligned)
        for t in tickers:
            assert t in result


class TestComparisonServiceDailyReturns:
    @pytest.fixture
    def svc(self, test_cfg):
        from app.services.comparison_service import ComparisonService
        from app.repositories.market_data_repo import MarketDataRepository
        repo = MarketDataRepository()
        return ComparisonService(repo, test_cfg)

    def test_returns_has_date_and_return_pct(self, svc):
        aligned = _make_aligned(["AAPL"])
        result = svc._daily_returns(aligned)
        assert "date" in result["AAPL"][0]
        assert "return_pct" in result["AAPL"][0]

    def test_first_return_is_zero(self, svc):
        """pct_change().fillna(0) should make first row 0."""
        aligned = _make_aligned(["AAPL"])
        result = svc._daily_returns(aligned)
        assert result["AAPL"][0]["return_pct"] == pytest.approx(0.0)
