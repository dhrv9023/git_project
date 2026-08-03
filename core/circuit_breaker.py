"""
core/circuit_breaker.py — Three-State Circuit Breaker

Production pattern: prevents cascading failures when an external dependency
(yfinance API, disk I/O) is degraded. Follows Fowler's circuit breaker pattern
used in Netflix Hystrix and Resilience4j.

State machine:
    ┌─────────────────────────────────────────────────────────┐
    │  CLOSED  ──(failure_rate > threshold)──▶  OPEN          │
    │    ▲                                        │            │
    │    │                          (reset_timeout elapsed)    │
    │    │                                        ▼            │
    │    └──(probe succeeds)────── HALF_OPEN               │
    └─────────────────────────────────────────────────────────┘

Complexity:
  State check:  O(1)
  Window scan:  O(1) amortized (deque with fixed maxlen)
  Memory:       O(window_size) per breaker instance
"""

import time
import threading
import logging
from collections import deque
from enum import Enum
from typing import Callable, Optional, Any

log = logging.getLogger(__name__)


class BreakerState(Enum):
    CLOSED    = "closed"      # Normal operation — calls pass through
    OPEN      = "open"        # Failure detected — calls blocked immediately
    HALF_OPEN = "half_open"   # Recovery probe — one test call allowed


class CircuitBreakerError(Exception):
    """Raised when a call is blocked by an OPEN circuit."""
    pass


class CircuitBreaker:
    """
    Thread-safe circuit breaker with sliding-window failure rate tracking.

    Args:
        name:              Identifier for logging/metrics
        failure_threshold: Fraction of calls that must fail to OPEN [0.0–1.0]
        window_size:       Number of recent calls to track
        reset_timeout_s:   Seconds in OPEN state before probing (HALF_OPEN)
        half_open_max:     Max concurrent probes allowed in HALF_OPEN state

    Big-O:
        call():   O(1) amortized (deque append/popleft)
        _rate():  O(1) (precomputed failure count in sliding window)

    Memory:  O(window_size) per instance
    """

    def __init__(self, name: str,
                 failure_threshold: float = 0.5,
                 window_size: int = 20,
                 reset_timeout_s: float = 30.0,
                 half_open_max: int = 1):
        self.name              = name
        self.failure_threshold = failure_threshold
        self.window_size       = window_size
        self.reset_timeout_s   = reset_timeout_s
        self.half_open_max     = half_open_max

        self._state            = BreakerState.CLOSED
        self._lock             = threading.RLock()
        # Sliding window: True = success, False = failure
        self._window: deque[bool] = deque(maxlen=window_size)
        self._failure_count    = 0           # tracked incrementally O(1)
        self._open_since: Optional[float] = None
        self._half_open_probes = 0
        self._total_calls      = 0
        self._total_failures   = 0
        self._total_blocked    = 0

    # ── State queries ─────────────────────────────────────────────────────────

    @property
    def state(self) -> BreakerState:
        with self._lock:
            return self._check_transition()

    def _check_transition(self) -> BreakerState:
        """Check if OPEN → HALF_OPEN transition is due. Call under lock."""
        if self._state == BreakerState.OPEN:
            elapsed = time.monotonic() - (self._open_since or 0)
            if elapsed >= self.reset_timeout_s:
                self._state = BreakerState.HALF_OPEN
                self._half_open_probes = 0
                log.info(f"[CB:{self.name}] OPEN → HALF_OPEN after {elapsed:.1f}s")
        return self._state

    def _failure_rate(self) -> float:
        """Current failure rate in the sliding window. O(1)."""
        n = len(self._window)
        return (self._failure_count / n) if n > 0 else 0.0

    # ── Main entry point ──────────────────────────────────────────────────────

    def call(self, fn: Callable, *args, **kwargs) -> Any:
        """
        Execute fn(*args, **kwargs) through the circuit breaker.

        Raises:
            CircuitBreakerError: if circuit is OPEN
            Any exception fn raises (recorded as failure)
        """
        with self._lock:
            state = self._check_transition()
            self._total_calls += 1

            if state == BreakerState.OPEN:
                self._total_blocked += 1
                raise CircuitBreakerError(
                    f"Circuit '{self.name}' is OPEN — call blocked. "
                    f"Retry after {self.reset_timeout_s}s."
                )

            if state == BreakerState.HALF_OPEN:
                if self._half_open_probes >= self.half_open_max:
                    self._total_blocked += 1
                    raise CircuitBreakerError(
                        f"Circuit '{self.name}' is HALF_OPEN — max probes reached."
                    )
                self._half_open_probes += 1

        # Execute outside the lock to avoid holding it during I/O
        try:
            result = fn(*args, **kwargs)
            self._record(success=True)
            return result
        except Exception as exc:
            self._record(success=False)
            raise

    def _record(self, success: bool):
        """Record outcome and update state. O(1) amortized."""
        with self._lock:
            # Evict oldest entry from window
            if len(self._window) == self.window_size:
                evicted = self._window[0]
                if not evicted:
                    self._failure_count -= 1

            # Append new entry
            self._window.append(success)
            if not success:
                self._failure_count += 1
                self._total_failures += 1

            state = self._state

            if state == BreakerState.CLOSED:
                rate = self._failure_rate()
                if len(self._window) >= self.window_size and rate >= self.failure_threshold:
                    self._state      = BreakerState.OPEN
                    self._open_since = time.monotonic()
                    log.warning(
                        f"[CB:{self.name}] CLOSED → OPEN "
                        f"(failure_rate={rate:.1%}, threshold={self.failure_threshold:.1%})"
                    )

            elif state == BreakerState.HALF_OPEN:
                if success:
                    # Probe succeeded: reset to CLOSED
                    self._state         = BreakerState.CLOSED
                    self._window.clear()
                    self._failure_count = 0
                    log.info(f"[CB:{self.name}] HALF_OPEN → CLOSED (probe succeeded)")
                else:
                    # Probe failed: back to OPEN
                    self._state      = BreakerState.OPEN
                    self._open_since = time.monotonic()
                    log.warning(f"[CB:{self.name}] HALF_OPEN → OPEN (probe failed)")

    # ── Observability ─────────────────────────────────────────────────────────

    def stats(self) -> dict:
        with self._lock:
            return {
                "name":            self.name,
                "state":           self._state.value,
                "failure_rate":    round(self._failure_rate(), 4),
                "window_size":     len(self._window),
                "total_calls":     self._total_calls,
                "total_failures":  self._total_failures,
                "total_blocked":   self._total_blocked,
                "reset_timeout_s": self.reset_timeout_s,
            }

    def reset(self):
        """Manually reset to CLOSED (operator override)."""
        with self._lock:
            self._state         = BreakerState.CLOSED
            self._window.clear()
            self._failure_count = 0
            self._open_since    = None
            log.info(f"[CB:{self.name}] Manually reset to CLOSED")


# ── Global registry of breakers ───────────────────────────────────────────────
_BREAKERS: dict[str, CircuitBreaker] = {}
_BREAKER_LOCK = threading.Lock()


def get_breaker(name: str, **kwargs) -> CircuitBreaker:
    """Get or create a named circuit breaker (singleton per name)."""
    with _BREAKER_LOCK:
        if name not in _BREAKERS:
            _BREAKERS[name] = CircuitBreaker(name, **kwargs)
        return _BREAKERS[name]


def all_breaker_stats() -> list:
    with _BREAKER_LOCK:
        return [b.stats() for b in _BREAKERS.values()]
