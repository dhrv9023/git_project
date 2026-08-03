"""
core/metrics.py — Prometheus-Compatible Metrics Collector

Production pattern: blind systems fail silently. This module provides
Prometheus-style metrics that can be scraped by Grafana or printed
as plaintext. Zero external dependencies.

Metric types implemented:
  Counter   — monotonically increasing integer (requests, errors)
  Histogram — latency distribution with configurable buckets
  Gauge     — current value (queue depth, cache size, memory)

Big-O:
  Counter.inc():       O(1)
  Histogram.observe(): O(B) where B = number of buckets (typically ≤ 15)
  Gauge.set():         O(1)
  format_prometheus(): O(N) where N = total registered metrics

Memory: O(N × B) where N = metric families, B = label combinations
"""

import time
import threading
import math
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Any

log = logging.getLogger(__name__)


# ── Counter ───────────────────────────────────────────────────────────────────

class Counter:
    """
    Monotonically increasing counter. Thread-safe.

    Complexity: O(1) per inc()
    Memory:     O(L) where L = unique label combinations
    """

    def __init__(self, name: str, description: str = "", labels: List[str] = None):
        self.name        = name
        self.description = description
        self._label_keys = labels or []
        self._values: Dict[tuple, int] = defaultdict(int)
        self._lock = threading.Lock()

    def inc(self, amount: int = 1, **label_values):
        key = tuple(label_values.get(k, "") for k in self._label_keys)
        with self._lock:
            self._values[key] += amount

    def get(self, **label_values) -> int:
        key = tuple(label_values.get(k, "") for k in self._label_keys)
        with self._lock:
            return self._values[key]

    def snapshot(self) -> Dict[tuple, int]:
        with self._lock:
            return dict(self._values)


# ── Histogram ─────────────────────────────────────────────────────────────────

class Histogram:
    """
    Latency/size distribution with configurable exponential buckets.

    Default buckets: [1ms, 5ms, 10ms, 25ms, 50ms, 100ms, 250ms, 500ms, 1s, 2.5s, 5s, 10s]

    Percentile approximation: O(B) where B = number of bucket boundaries.
    For P99 estimation, walk buckets until cumulative count ≥ 0.99 × total.

    Memory: O(B × L) where L = unique label combinations
    """

    DEFAULT_BUCKETS_S = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]

    def __init__(self, name: str, description: str = "",
                 buckets: List[float] = None, labels: List[str] = None):
        self.name        = name
        self.description = description
        self._buckets    = sorted(buckets or self.DEFAULT_BUCKETS_S) + [math.inf]
        self._label_keys = labels or []
        # Per label-combo: list of bucket counts + sum + count
        self._data: Dict[tuple, dict] = {}
        self._lock = threading.Lock()

    def _init_label(self, key: tuple):
        self._data[key] = {
            "buckets": [0] * len(self._buckets),  # cumulative counts per boundary
            "sum":     0.0,
            "count":   0,
        }

    def observe(self, value: float, **label_values):
        key = tuple(label_values.get(k, "") for k in self._label_keys)
        with self._lock:
            if key not in self._data:
                self._init_label(key)
            d = self._data[key]
            d["sum"]   += value
            d["count"] += 1
            for i, boundary in enumerate(self._buckets):  # O(B)
                if value <= boundary:
                    d["buckets"][i] += 1

    def percentile(self, p: float, **label_values) -> Optional[float]:
        """Approximate Pxx from bucket counts. O(B)."""
        key = tuple(label_values.get(k, "") for k in self._label_keys)
        with self._lock:
            if key not in self._data:
                return None
            d = self._data[key]
            total = d["count"]
            if total == 0:
                return None
            target = p * total
            for i, boundary in enumerate(self._buckets):
                if d["buckets"][i] >= target:
                    return boundary
            return self._buckets[-2]   # last finite bucket

    def snapshot(self, **label_values) -> dict:
        key = tuple(label_values.get(k, "") for k in self._label_keys)
        with self._lock:
            if key not in self._data:
                return {}
            d = self._data[key]
            total = d["count"]
            return {
                "count": total,
                "sum":   round(d["sum"], 6),
                "mean":  round(d["sum"] / total, 6) if total else 0,
                "p50":   self.percentile(0.50, **label_values),
                "p95":   self.percentile(0.95, **label_values),
                "p99":   self.percentile(0.99, **label_values),
                "buckets": {
                    f"le_{b}": d["buckets"][i]
                    for i, b in enumerate(self._buckets)
                    if not math.isinf(b)
                },
            }


# ── Gauge ─────────────────────────────────────────────────────────────────────

class Gauge:
    """
    Current numeric value. Can go up or down. Thread-safe.
    Complexity: O(1) per set/inc/dec
    """

    def __init__(self, name: str, description: str = ""):
        self.name        = name
        self.description = description
        self._value      = 0.0
        self._lock       = threading.Lock()

    def set(self, v: float):
        with self._lock:
            self._value = v

    def inc(self, v: float = 1.0):
        with self._lock:
            self._value += v

    def dec(self, v: float = 1.0):
        with self._lock:
            self._value -= v

    def get(self) -> float:
        with self._lock:
            return self._value


# ── Timer context manager ─────────────────────────────────────────────────────

class Timer:
    """Context manager that records elapsed time into a Histogram."""

    def __init__(self, histogram: Histogram, **label_values):
        self._hist   = histogram
        self._labels = label_values
        self._start  = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        elapsed = time.perf_counter() - self._start
        self._hist.observe(elapsed, **self._labels)


# ── Registry ──────────────────────────────────────────────────────────────────

class MetricsRegistry:
    """
    Central registry for all metrics. Provides Prometheus text format export.
    Singleton pattern — use the module-level `REGISTRY` instance.
    """

    def __init__(self):
        self._metrics: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def register(self, metric) -> Any:
        with self._lock:
            self._metrics[metric.name] = metric
        return metric

    def counter(self, name: str, description: str = "", labels: List[str] = None) -> Counter:
        m = Counter(name, description, labels)
        return self.register(m)

    def histogram(self, name: str, description: str = "", labels: List[str] = None) -> Histogram:
        m = Histogram(name, description, labels=labels)
        return self.register(m)

    def gauge(self, name: str, description: str = "") -> Gauge:
        m = Gauge(name, description)
        return self.register(m)

    def format_prometheus(self) -> str:
        """
        Export all metrics in Prometheus exposition text format.
        Complexity: O(N × L) where N = metrics, L = label combinations
        """
        lines = []
        with self._lock:
            metrics = dict(self._metrics)

        for name, m in metrics.items():
            lines.append(f"# HELP {name} {m.description}")
            if isinstance(m, Counter):
                lines.append(f"# TYPE {name} counter")
                for labels, val in m.snapshot().items():
                    label_str = _fmt_labels(m._label_keys, labels)
                    lines.append(f"{name}{label_str} {val}")
            elif isinstance(m, Gauge):
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name} {m.get()}")
            elif isinstance(m, Histogram):
                lines.append(f"# TYPE {name} histogram")
                # summary format for simplicity
                for key in m._data:
                    snap = m.snapshot(**dict(zip(m._label_keys, key)))
                    label_str = _fmt_labels(m._label_keys, key)
                    if snap:
                        lines.append(f"{name}_count{label_str} {snap['count']}")
                        lines.append(f"{name}_sum{label_str} {snap['sum']}")
                        for q, p in [("0.5", "p50"), ("0.95", "p95"), ("0.99", "p99")]:
                            v = snap.get(p)
                            if v is not None:
                                ql = label_str.rstrip("}") + f',quantile="{q}"' + "}"
                                if ql.startswith("}"):
                                    ql = '{' + ql[1:]
                                lines.append(f"{name}{ql} {v}")
        return "\n".join(lines) + "\n"

    def as_dict(self) -> dict:
        """JSON-serializable snapshot of all metrics."""
        result = {}
        with self._lock:
            for name, m in self._metrics.items():
                if isinstance(m, Counter):
                    result[name] = {"type": "counter", "values": {str(k): v for k, v in m.snapshot().items()}}
                elif isinstance(m, Gauge):
                    result[name] = {"type": "gauge", "value": m.get()}
                elif isinstance(m, Histogram):
                    snaps = {}
                    for key in m._data:
                        snaps[str(key)] = m.snapshot(**dict(zip(m._label_keys, key)))
                    result[name] = {"type": "histogram", "snapshots": snaps}
        return result


def _fmt_labels(keys: list, values: tuple) -> str:
    if not keys:
        return ""
    pairs = ",".join(f'{k}="{v}"' for k, v in zip(keys, values))
    return "{" + pairs + "}"


# ── Module-level singleton ────────────────────────────────────────────────────
REGISTRY = MetricsRegistry()

# Pre-registered application metrics
http_requests_total    = REGISTRY.counter("http_requests_total", "Total HTTP requests", labels=["method", "endpoint", "status"])
http_latency_seconds   = REGISTRY.histogram("http_latency_seconds", "HTTP handler latency", labels=["endpoint"])
training_jobs_total    = REGISTRY.counter("training_jobs_total", "Total training jobs", labels=["status"])
training_duration_s    = REGISTRY.histogram("training_duration_seconds", "Model training duration")
inference_latency_s    = REGISTRY.histogram("inference_latency_seconds", "Inference latency", labels=["cache_layer"])
cache_hits_total       = REGISTRY.counter("cache_hits_total", "Cache hits", labels=["layer"])
cache_misses_total     = REGISTRY.counter("cache_misses_total", "Cache misses")
yfinance_calls_total   = REGISTRY.counter("yfinance_calls_total", "yfinance API calls", labels=["status"])
circuit_breaker_opens  = REGISTRY.counter("circuit_breaker_opens_total", "Times a circuit opened", labels=["name"])
queue_depth            = REGISTRY.gauge("job_queue_depth", "Current job queue depth")
dlq_depth              = REGISTRY.gauge("dlq_depth", "Dead-letter queue depth")
active_workers         = REGISTRY.gauge("active_worker_threads", "Active training worker threads")
memory_cache_entries   = REGISTRY.gauge("memory_cache_entries", "Entries in memory cache")
disk_cache_entries     = REGISTRY.gauge("disk_cache_entries", "Entries in disk cache")
model_versions_total   = REGISTRY.gauge("model_versions_total", "Total model versions in registry")
