"""
core/rate_limiter.py — Token Bucket + Sliding Window Rate Limiter

Two algorithms implemented:

1. TokenBucketLimiter  — smooth burst handling, O(1) per check
   Tokens refill continuously at `rate` tokens/second.
   A burst of up to `capacity` tokens is allowed.
   Classic algorithm used by AWS API Gateway and GCP Cloud Endpoints.

2. SlidingWindowLimiter — strict per-window counting, O(1) amortized
   Uses a deque of timestamps. Evicts expired entries on each check.
   Guarantees at most `max_calls` in any rolling `window_s` window.

Memory complexity:
  TokenBucket:    O(1) per key
  SlidingWindow:  O(max_calls) per key (deque bounded by max_calls)

Throughput:
  Both: O(1) per request check under lock
"""

import time
import threading
import logging
from collections import deque
from typing import Dict

log = logging.getLogger(__name__)


# ── Token Bucket ──────────────────────────────────────────────────────────────

class TokenBucket:
    """
    Single key token bucket.

    Tokens = capacity at start. Each call consumes 1 token.
    Tokens refill at `rate` per second up to `capacity`.
    """

    def __init__(self, capacity: float, rate: float):
        """
        Args:
            capacity: max burst (tokens)
            rate:     refill rate (tokens/second)
        """
        self.capacity  = float(capacity)
        self.rate      = float(rate)
        self._tokens   = float(capacity)
        self._last     = time.monotonic()
        self._lock     = threading.Lock()
        self._rejected = 0
        self._allowed  = 0

    def allow(self, consume: float = 1.0) -> bool:
        """Return True if `consume` tokens available, deducting them."""
        with self._lock:
            now     = time.monotonic()
            elapsed = now - self._last
            self._last = now
            # Refill
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
            if self._tokens >= consume:
                self._tokens -= consume
                self._allowed += 1
                return True
            self._rejected += 1
            return False

    def stats(self) -> dict:
        with self._lock:
            return {
                "tokens_available": round(self._tokens, 2),
                "capacity":         self.capacity,
                "rate_per_s":       self.rate,
                "allowed":          self._allowed,
                "rejected":         self._rejected,
            }


class TokenBucketLimiter:
    """
    Multi-key token bucket rate limiter.
    Key is typically an IP address or user ID.

    Memory: O(N) where N = number of unique keys seen
    """

    def __init__(self, capacity: float = 10.0, rate: float = 2.0):
        self.capacity  = capacity
        self.rate      = rate
        self._buckets: Dict[str, TokenBucket] = {}
        self._lock     = threading.Lock()

    def allow(self, key: str = "default", consume: float = 1.0) -> bool:
        with self._lock:
            if key not in self._buckets:
                self._buckets[key] = TokenBucket(self.capacity, self.rate)
        return self._buckets[key].allow(consume)

    def stats(self) -> dict:
        with self._lock:
            return {k: v.stats() for k, v in self._buckets.items()}


# ── Sliding Window ────────────────────────────────────────────────────────────

class SlidingWindowLimiter:
    """
    Strict sliding window rate limiter.
    Guarantees at most `max_calls` in any rolling `window_s` seconds.

    Memory: O(max_calls) per key (deque bounded by max_calls)
    Complexity: O(K) per check where K = expired entries to evict (amortized O(1))
    """

    def __init__(self, max_calls: int = 60, window_s: float = 60.0):
        self.max_calls = max_calls
        self.window_s  = window_s
        self._windows: Dict[str, deque] = {}
        self._lock     = threading.Lock()
        self._rejected: Dict[str, int]  = {}

    def allow(self, key: str = "default") -> bool:
        now = time.monotonic()
        with self._lock:
            if key not in self._windows:
                self._windows[key]  = deque()
                self._rejected[key] = 0

            win = self._windows[key]
            cutoff = now - self.window_s

            # Evict expired — amortized O(1) per call over long runs
            while win and win[0] < cutoff:
                win.popleft()

            if len(win) < self.max_calls:
                win.append(now)
                return True

            self._rejected[key] += 1
            return False

    def stats(self) -> dict:
        with self._lock:
            return {
                k: {
                    "calls_in_window": len(self._windows[k]),
                    "max_calls":       self.max_calls,
                    "window_s":        self.window_s,
                    "rejected":        self._rejected.get(k, 0),
                }
                for k in self._windows
            }


# ── Combined limiter used by Flask middleware ──────────────────────────────────

class RateLimiter:
    """
    Composite limiter: token bucket (burst) + sliding window (strict).
    Both must allow for the call to proceed.
    """

    def __init__(self, burst_capacity: float = 20.0,
                 burst_rate: float = 5.0,
                 window_max: int = 30,
                 window_s: float = 60.0):
        self._tb  = TokenBucketLimiter(capacity=burst_capacity, rate=burst_rate)
        self._sw  = SlidingWindowLimiter(max_calls=window_max, window_s=window_s)

    def allow(self, key: str = "default") -> bool:
        return self._tb.allow(key) and self._sw.allow(key)

    def stats(self) -> dict:
        return {"token_bucket": self._tb.stats(), "sliding_window": self._sw.stats()}
