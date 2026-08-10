"""
tests/unit/test_exceptions.py — Unit tests for domain exception hierarchy.
"""
import pytest
from app.domain.exceptions import (
    StockBuddyError, DataFetchError, InsufficientDataError,
    ModelNotFoundError, TrainingError, ConfigurationError, CacheError,
)


class TestExceptionHierarchy:
    def test_all_subclass_stockbuddy_error(self):
        for exc_class in [DataFetchError, InsufficientDataError, ModelNotFoundError,
                          TrainingError, ConfigurationError, CacheError]:
            assert issubclass(exc_class, StockBuddyError)

    def test_all_subclass_exception(self):
        assert issubclass(StockBuddyError, Exception)

    def test_can_catch_all_with_base(self):
        caught = []
        for exc_class in [DataFetchError, ModelNotFoundError, TrainingError]:
            try:
                raise exc_class("test")
            except StockBuddyError as e:
                caught.append(e)
        assert len(caught) == 3


class TestInsufficientDataError:
    def test_message_contains_counts(self):
        exc = InsufficientDataError(required=200, available=50)
        assert "200" in exc.message
        assert "50" in exc.message

    def test_attributes_set(self):
        exc = InsufficientDataError(required=100, available=30)
        assert exc.required == 100
        assert exc.available == 30


class TestModelNotFoundError:
    def test_message_contains_ticker(self):
        exc = ModelNotFoundError(ticker="TSLA", version="best")
        assert "TSLA" in exc.message

    def test_to_dict_has_error_key(self):
        exc = ModelNotFoundError(ticker="AAPL")
        d = exc.to_dict()
        assert "error" in d


class TestStockBuddyErrorToDict:
    def test_message_key(self):
        exc = DataFetchError("download failed", detail="timeout")
        d = exc.to_dict()
        assert d["error"] == "download failed"
        assert d["detail"] == "timeout"

    def test_no_detail_omits_key(self):
        exc = DataFetchError("simple error")
        d = exc.to_dict()
        assert "detail" not in d
