"""
ml/batch_predictor.py — Batched Async Inference Engine

Problem: Each incoming /api/v2/predict call runs model.predict() independently.
For N simultaneous requests, this is N separate forward passes — wasted GPU/CPU
parallelism since neural network predict() is vectorized and runs a batch of size
B at negligible extra cost vs a batch of size 1.

Solution: Collect individual prediction requests into micro-batches and run them
together. This is the pattern used by TensorFlow Serving, TorchServe, and
NVIDIA Triton Inference Server.

Architecture:
  Client A ──┐
  Client B ──┼──▶ RequestQueue ──▶ Batcher Thread ──▶ model.predict(batch)
  Client C ──┘         │                                      │
       ▲               │                                      │
       │               └──(block on Future)                   │
       └─────────────────────────── resolve Future ───────────┘

Throughput gains:
  Sequential:  N requests × predict_time(1)   = N × T
  Batched:     ceil(N/B)  × predict_time(B)   ≈ N × T/B  (for B ≤ GPU capacity)
  Speedup:     B× on GPU, ~2-3× on CPU (memory bandwidth savings)

Latency impact:
  Added latency = max_wait_ms = 50ms (configurable)
  Acceptable for async API where P99 matters more than P50

Big-O:
  enqueue():  O(1) — queue append
  _batch():   O(B) — zip results to futures
  predict():  O(S×F×B) — neural net forward pass (same as N individual calls
              but amortized over batch)

Memory:
  O(B × S × F) per batch (S=seq_len, F=features, B=batch_size)
  Released after predict() completes
"""

import time
import queue
import logging
import threading
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Any
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class PredictionRequest:
    """One request in the batch queue."""
    sequence:    np.ndarray        # shape: (seq_len, n_features)
    future:      Future            # resolved with prediction result
    enqueued_at: float = field(default_factory=time.perf_counter)


class BatchPredictor:
    """
    Collects prediction requests and runs them in micro-batches.

    Each BatchPredictor wraps a single Keras model.
    Multiple models (LSTM, GRU, Transformer) each get their own BatchPredictor
    and their outputs are averaged by the caller.

    Args:
        model:         Keras model with a predict() method
        model_name:    For logging
        max_batch:     Maximum sequences per forward pass
        max_wait_ms:   Maximum time to wait for a full batch (milliseconds)
    """

    def __init__(self, model, model_name: str = "model",
                 max_batch: int = 32, max_wait_ms: float = 50.0):
        self.model        = model
        self.model_name   = model_name
        self.max_batch    = max_batch
        self.max_wait_ms  = max_wait_ms / 1000.0   # convert to seconds

        self._queue:  queue.Queue[PredictionRequest] = queue.Queue()
        self._stop    = threading.Event()
        self._thread  = threading.Thread(target=self._batch_loop,
                                         name=f"sb_batch_{model_name}",
                                         daemon=True)
        self._thread.start()

        # Metrics
        self._n_batches    = 0
        self._n_requests   = 0
        self._total_wait_s = 0.0

    def predict_async(self, sequence: np.ndarray) -> Future:
        """
        Submit one sequence for prediction. Returns a Future.
        The future resolves to a scalar predicted log-return (float).

        Non-blocking — O(1).
        """
        f = Future()
        f.set_running_or_notify_cancel()
        req = PredictionRequest(sequence=sequence, future=f)
        self._queue.put(req)
        return f

    def predict_sync(self, sequence: np.ndarray,
                     timeout_s: float = 10.0) -> float:
        """Submit and block until result. Convenience wrapper."""
        return self.predict_async(sequence).result(timeout=timeout_s)

    def _batch_loop(self):
        """
        Daemon thread: collect requests into batches and run predict().
        Exits when stop() is called and queue is drained.
        """
        while not self._stop.is_set() or not self._queue.empty():
            batch = self._collect_batch()
            if not batch:
                continue
            self._run_batch(batch)

    def _collect_batch(self) -> List[PredictionRequest]:
        """
        Collect up to max_batch requests, waiting at most max_wait_ms.

        Strategy: block on first item (up to max_wait_ms), then drain
        non-blocking until batch is full or queue is empty.

        Returns empty list if no requests arrive within timeout.
        """
        batch = []
        deadline = time.perf_counter() + self.max_wait_ms

        # Wait for at least one item
        try:
            remaining = deadline - time.perf_counter()
            req = self._queue.get(timeout=max(0.001, remaining))
            batch.append(req)
        except queue.Empty:
            return []

        # Drain non-blocking until max_batch or deadline
        while len(batch) < self.max_batch and time.perf_counter() < deadline:
            try:
                req = self._queue.get_nowait()
                batch.append(req)
            except queue.Empty:
                break

        return batch

    def _run_batch(self, batch: List[PredictionRequest]):
        """
        Run model.predict() on the full batch. O(B × S × F).
        Resolves each Future with its corresponding prediction.
        """
        self._n_batches  += 1
        self._n_requests += len(batch)

        # Stack sequences into (B, seq_len, n_features) array
        sequences = np.stack([r.sequence for r in batch], axis=0).astype(np.float32)

        # Measure wait time for the oldest request in batch
        oldest_wait = time.perf_counter() - min(r.enqueued_at for r in batch)
        self._total_wait_s += oldest_wait

        try:
            # Single batched forward pass
            predictions = self.model.predict(sequences, verbose=0, batch_size=len(batch))
            # predictions shape: (B, 1) or (B,)
            predictions = predictions.ravel()

            for req, pred in zip(batch, predictions):
                try:
                    req.future.set_result(float(pred))
                except Exception:
                    pass   # Future may already be cancelled

        except Exception as exc:
            log.error(f"[Batch:{self.model_name}] predict failed: {exc}")
            for req in batch:
                try:
                    req.future.set_exception(exc)
                except Exception:
                    pass

    def stats(self) -> dict:
        qsize = self._queue.qsize()
        avg_wait = (self._total_wait_s / self._n_batches
                    if self._n_batches > 0 else 0.0)
        avg_batch = (self._n_requests / self._n_batches
                     if self._n_batches > 0 else 0.0)
        return {
            "model":            self.model_name,
            "queue_depth":      qsize,
            "total_batches":    self._n_batches,
            "total_requests":   self._n_requests,
            "avg_batch_size":   round(avg_batch, 2),
            "avg_wait_ms":      round(avg_wait * 1000, 2),
            "max_batch":        self.max_batch,
            "max_wait_ms":      self.max_wait_ms * 1000,
        }

    def stop(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5)


class EnsembleBatchPredictor:
    """
    Wraps multiple BatchPredictor instances (one per model arch).
    Runs predictions in parallel across models and averages results.

    Throughput: K models × B requests per batch — K parallel forward passes
    Latency:    max(LSTM_latency, GRU_latency, Transformer_latency)
                (parallel, not sequential)
    """

    def __init__(self, models_dict: dict, max_batch: int = 32, max_wait_ms: float = 50.0):
        self._predictors = {
            name: BatchPredictor(model, model_name=name,
                                 max_batch=max_batch, max_wait_ms=max_wait_ms)
            for name, model in models_dict.items()
            if not name.endswith("_history")
        }

    def predict_sync(self, sequence: np.ndarray, timeout_s: float = 30.0) -> float:
        """
        Submit sequence to all models in parallel, return ensemble mean.
        Latency = max(individual model latencies) + queue wait
        """
        futures = {
            name: pred.predict_async(sequence)
            for name, pred in self._predictors.items()
        }
        results = []
        for name, f in futures.items():
            try:
                results.append(f.result(timeout=timeout_s))
            except Exception as e:
                log.warning(f"Model {name} prediction failed: {e}")

        if not results:
            raise RuntimeError("All models failed in ensemble prediction")
        return float(np.mean(results))

    def stats(self) -> dict:
        return {name: pred.stats() for name, pred in self._predictors.items()}

    def stop(self):
        for pred in self._predictors.values():
            pred.stop()
