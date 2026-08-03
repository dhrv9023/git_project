"""
tests/load_test.py — Throughput & Latency Benchmark

Benchmark strategy:
  1. Warmup:   10 requests to ensure models are loaded and cache is warm
  2. Ramp-up:  Gradually increase concurrency from 1 → max_workers
  3. Sustained: Run at target concurrency for duration_s seconds
  4. Cool-down: Drain in-flight requests

Metrics collected:
  - Throughput (requests/second)
  - Latency percentiles: P50, P95, P99
  - Error rate
  - Cache hit rate

Usage:
  # Run against a live server
  python tests/load_test.py --host http://localhost:5000 --workers 10 --duration 30

  # Benchmark unit tests (no server needed)
  python tests/load_test.py --unit
"""

import argparse
import math
import random
import statistics
import sys
import time
import threading
import urllib.request
import urllib.error
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Dict, Optional


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class RequestResult:
    endpoint:    str
    latency_s:   float
    status_code: int
    success:     bool
    from_cache:  bool = False
    error:       Optional[str] = None


# ── HTTP helper ───────────────────────────────────────────────────────────────

def _post(base_url: str, path: str, payload: dict, timeout: float = 30.0) -> RequestResult:
    start = time.perf_counter()
    url   = f"{base_url}{path}"
    data  = json.dumps(payload).encode()
    req   = urllib.request.Request(url, data=data,
                                   headers={"Content-Type": "application/json"},
                                   method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read())
            latency = time.perf_counter() - start
            return RequestResult(
                endpoint    = path,
                latency_s   = latency,
                status_code = resp.status,
                success     = True,
                from_cache  = body.get("from_cache", False),
            )
    except urllib.error.HTTPError as e:
        return RequestResult(path, time.perf_counter()-start, e.code, False, error=str(e))
    except Exception as e:
        return RequestResult(path, time.perf_counter()-start, 0, False, error=str(e))


def _get(base_url: str, path: str, timeout: float = 10.0) -> RequestResult:
    start = time.perf_counter()
    url   = f"{base_url}{path}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return RequestResult(path, time.perf_counter()-start, resp.status, True)
    except Exception as e:
        return RequestResult(path, time.perf_counter()-start, 0, False, error=str(e))


# ── Benchmark scenarios ───────────────────────────────────────────────────────

TICKERS = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]

def scenario_health(base: str) -> RequestResult:
    return _get(base, "/api/health")

def scenario_v2_metrics(base: str) -> RequestResult:
    return _get(base, "/api/v2/metrics")

def scenario_v3_queue(base: str) -> RequestResult:
    return _get(base, "/api/v3/queue")

def scenario_predict_cached(base: str) -> RequestResult:
    """Hit /api/v2/predict with a common ticker — tests cache hit rate."""
    return _post(base, "/api/v2/predict", {
        "ticker":     "AAPL",
        "start_date": "2021-01-01",
        "end_date":   "2024-01-01",
        "version":    "best",
    })

def scenario_regime(base: str) -> RequestResult:
    ticker = random.choice(TICKERS)
    return _post(base, "/api/regime", {"ticker": ticker})


# ── Load runner ───────────────────────────────────────────────────────────────

def run_load_test(base_url: str, scenario, workers: int = 5,
                  duration_s: float = 30.0, warmup_n: int = 3) -> Dict:
    """
    Run scenario() concurrently with `workers` threads for `duration_s` seconds.

    Returns dict with:
      - throughput_rps:  requests per second
      - p50/p95/p99:     latency percentiles (seconds)
      - error_rate:      fraction of failed requests
      - cache_hit_rate:  fraction of responses served from cache
      - total_requests:  total attempts
    """
    results: List[RequestResult] = []
    results_lock = threading.Lock()
    stop_event   = threading.Event()

    def worker():
        while not stop_event.is_set():
            r = scenario(base_url)
            with results_lock:
                results.append(r)

    # Warmup
    print(f"  Warming up ({warmup_n} requests)...", flush=True)
    with ThreadPoolExecutor(max_workers=min(warmup_n, workers)) as pool:
        warmup_futures = [pool.submit(scenario, base_url) for _ in range(warmup_n)]
        for f in warmup_futures:
            try: f.result(timeout=60)
            except Exception: pass

    # Sustained load
    print(f"  Running {workers} workers for {duration_s}s...", flush=True)
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(workers)]
    t_start = time.perf_counter()
    for t in threads:
        t.start()

    time.sleep(duration_s)
    stop_event.set()

    for t in threads:
        t.join(timeout=5)

    elapsed = time.perf_counter() - t_start

    if not results:
        return {"error": "No results collected"}

    latencies = [r.latency_s for r in results]
    errors    = [r for r in results if not r.success]
    cached    = [r for r in results if r.from_cache]
    latencies.sort()

    def pct(p):
        idx = max(0, min(len(latencies)-1, int(math.ceil(p * len(latencies))) - 1))
        return round(latencies[idx] * 1000, 2)  # ms

    return {
        "total_requests":  len(results),
        "throughput_rps":  round(len(results) / elapsed, 2),
        "elapsed_s":       round(elapsed, 2),
        "workers":         workers,
        "error_rate":      round(len(errors) / len(results), 4),
        "error_count":     len(errors),
        "cache_hit_rate":  round(len(cached) / len(results), 4) if results else 0,
        "latency_ms": {
            "min":  round(min(latencies)*1000, 2),
            "p50":  pct(0.50),
            "p95":  pct(0.95),
            "p99":  pct(0.99),
            "max":  round(max(latencies)*1000, 2),
            "mean": round(statistics.mean(latencies)*1000, 2),
        },
    }


# ── Unit benchmarks (no server needed) ───────────────────────────────────────

def bench_circuit_breaker():
    """Measure circuit breaker overhead per call."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from core.circuit_breaker import CircuitBreaker

    cb = CircuitBreaker("test", failure_threshold=0.5, window_size=100, reset_timeout_s=1.0)
    fn = lambda: 42

    N = 100_000
    t0 = time.perf_counter()
    for _ in range(N):
        try:
            cb.call(fn)
        except Exception:
            pass
    elapsed = time.perf_counter() - t0

    ns_per_call = elapsed / N * 1e9
    return {
        "test":          "circuit_breaker.call()",
        "n":             N,
        "total_s":       round(elapsed, 4),
        "ns_per_call":   round(ns_per_call, 1),
        "throughput_M":  round(N / elapsed / 1e6, 2),
        "verdict":       "OK" if ns_per_call < 5000 else "SLOW",
    }


def bench_rate_limiter():
    """Measure token bucket overhead per check."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from core.rate_limiter import TokenBucketLimiter

    limiter = TokenBucketLimiter(capacity=1e9, rate=1e9)  # effectively unlimited
    N = 100_000
    t0 = time.perf_counter()
    for _ in range(N):
        limiter.allow("test_key")
    elapsed = time.perf_counter() - t0

    ns_per_call = elapsed / N * 1e9
    return {
        "test":          "TokenBucketLimiter.allow()",
        "n":             N,
        "total_s":       round(elapsed, 4),
        "ns_per_call":   round(ns_per_call, 1),
        "throughput_M":  round(N / elapsed / 1e6, 2),
        "verdict":       "OK" if ns_per_call < 5000 else "SLOW",
    }


def bench_metrics_counter():
    """Measure metrics counter overhead per increment."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from core.metrics import REGISTRY

    ctr = REGISTRY.counter("bench_test_counter", labels=["endpoint"])
    N = 1_000_000
    t0 = time.perf_counter()
    for _ in range(N):
        ctr.inc(endpoint="/api/test")
    elapsed = time.perf_counter() - t0

    ns_per_call = elapsed / N * 1e9
    return {
        "test":          "Counter.inc()",
        "n":             N,
        "total_s":       round(elapsed, 4),
        "ns_per_call":   round(ns_per_call, 1),
        "throughput_M":  round(N / elapsed / 1e6, 2),
        "verdict":       "OK" if ns_per_call < 1000 else "SLOW",
    }


def bench_priority_queue():
    """Measure priority queue submit/pop overhead."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from ml.queue import PriorityJobQueue, PRIORITY_NORMAL

    q = PriorityJobQueue(max_workers=0)  # 0 workers: just benchmark the queue
    fn = lambda: None
    N = 10_000

    t0 = time.perf_counter()
    for i in range(N):
        q.submit(fn, payload={"ticker": "AAPL"}, priority=PRIORITY_NORMAL)
    elapsed = time.perf_counter() - t0

    ns_per_call = elapsed / N * 1e9
    q.shutdown()

    return {
        "test":          "PriorityJobQueue.submit() [heap push O(log N)]",
        "n":             N,
        "total_s":       round(elapsed, 4),
        "ns_per_call":   round(ns_per_call, 1),
        "throughput_M":  round(N / elapsed / 1e6, 3),
        "verdict":       "OK" if ns_per_call < 50_000 else "SLOW",
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def print_result(label: str, r: dict):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    for k, v in r.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk:12s}: {vv}")
        else:
            print(f"  {k:20s}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StockBuddy Load Test")
    parser.add_argument("--host",     default="http://localhost:5000")
    parser.add_argument("--workers",  type=int,   default=5)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--unit",     action="store_true",
                        help="Run unit benchmarks only (no server needed)")
    args = parser.parse_args()

    if args.unit:
        print("\n[Unit Benchmarks — no server required]\n")
        for bench_fn in [bench_circuit_breaker, bench_rate_limiter,
                         bench_metrics_counter, bench_priority_queue]:
            try:
                result = bench_fn()
                print_result(result["test"], result)
            except Exception as e:
                print(f"\n  SKIP: {bench_fn.__name__} — {e}")
        sys.exit(0)

    # Integration load tests
    scenarios = [
        ("Health Check",        scenario_health),
        ("v2 Metrics",          scenario_v2_metrics),
        ("v3 Queue",            scenario_v3_queue),
        ("Cached Prediction",   scenario_predict_cached),
    ]

    print(f"\nLoad testing {args.host} | workers={args.workers} | duration={args.duration}s\n")

    for name, scn in scenarios:
        print(f"[{name}]")
        try:
            result = run_load_test(args.host, scn,
                                   workers=args.workers,
                                   duration_s=args.duration)
            print_result(name, result)
        except Exception as e:
            print(f"  ERROR: {e}")
        print()
