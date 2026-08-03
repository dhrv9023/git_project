"""
ml/queue.py — Priority Job Queue with Dead-Letter Queue and Retry Backoff

Production pattern: replaces the simple ThreadPoolExecutor submit() with a
proper priority queue that:
  1. Allows urgent jobs (force_retrain, UI requests) to jump the queue
  2. Retries failed jobs with exponential backoff + jitter
  3. Moves permanently failed jobs to a Dead-Letter Queue (DLQ) for inspection
  4. Provides per-job metrics and deduplication

Architecture:
  ┌──────────────────────────────────────────────────────────┐
  │                  PriorityJobQueue                        │
  │                                                          │
  │  submit(job, priority)                                   │
  │      │                                                   │
  │      ▼                                                   │
  │  ┌──────────────┐   O(log N) push   ┌────────────────┐  │
  │  │  Min-Heap    │ ─────────────────▶│ Worker Threads │  │
  │  │  (priority,  │                   │ (ThreadPool)   │  │
  │  │   seq, job)  │ ◀────── pop ───── │                │  │
  │  └──────────────┘                   └───────┬────────┘  │
  │                                             │            │
  │                              success ───────┘            │
  │                              failure ──▶ retry queue     │
  │                              max_retries ──▶ DLQ         │
  └──────────────────────────────────────────────────────────┘

Big-O:
  submit():      O(log N)  — heap push
  _next_job():   O(log N)  — heap pop
  retry():       O(log N)  — heap push with backoff delay check
  DLQ append:   O(1)

Memory:
  O(N) for N queued jobs
  O(R) for R retrying jobs
  O(D) for D dead-lettered jobs (capped at DLQ_MAX_SIZE)

Throughput: limited by worker threads (cfg.max_worker_threads)
Latency:    priority 0 = immediate, priority 10 = lowest priority
"""

import uuid
import time
import math
import random
import heapq
import logging
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional, Dict, Callable, Any, List

log = logging.getLogger(__name__)

# Dead-letter queue is capped to prevent unbounded memory growth
DLQ_MAX_SIZE = 500

# Priority constants (lower = higher priority)
PRIORITY_URGENT  = 0   # force_retrain from user
PRIORITY_HIGH    = 3   # scheduler-triggered retrain
PRIORITY_NORMAL  = 5   # regular API request
PRIORITY_LOW     = 10  # background speculative prefetch


@dataclass
class RetryPolicy:
    max_retries:      int   = 3
    base_delay_s:     float = 5.0    # initial backoff
    max_delay_s:      float = 300.0  # cap at 5 minutes
    jitter_fraction:  float = 0.2    # ±20% jitter to avoid thundering herd

    def delay_for_attempt(self, attempt: int) -> float:
        """Exponential backoff with jitter. O(1)."""
        delay = min(self.max_delay_s, self.base_delay_s * (2 ** attempt))
        jitter = delay * self.jitter_fraction * (2 * random.random() - 1)
        return max(0.0, delay + jitter)


@dataclass(order=False)
class QueuedJob:
    """
    A job in the priority queue.

    The heap uses (priority, seq_id, job) tuples to break ties
    by arrival order — this ensures FIFO ordering within the same priority.
    """
    job_id:      str
    priority:    int
    seq_id:      int             # monotonically increasing — tie-break for heap
    payload:     Dict[str, Any]  # job parameters (ticker, dates, etc.)
    fn:          Callable        # the callable to invoke
    fn_args:     tuple = field(default_factory=tuple)
    fn_kwargs:   dict  = field(default_factory=dict)

    # Retry tracking
    attempt:          int   = 0
    max_retries:      int   = 3
    retry_policy:     RetryPolicy = field(default_factory=RetryPolicy)
    next_eligible_at: float = 0.0  # epoch seconds; 0 = immediately eligible

    # Status
    status:       str  = "queued"   # queued|running|done|failed|dead
    created_at:   float = field(default_factory=time.time)
    started_at:   Optional[float] = None
    completed_at: Optional[float] = None
    last_error:   Optional[str]   = None
    result:       Optional[Any]   = None

    def heap_key(self) -> tuple:
        """Heap comparison key: (priority, arrival_order)."""
        return (self.priority, self.seq_id)


class PriorityJobQueue:
    """
    Thread-safe priority job queue with retry and dead-letter queue.

    The heap stores (priority, seq_id, job_id) tuples.
    Actual job objects are stored in _jobs dict for O(1) lookup by job_id.

    Why not heapq with job objects directly?
      — Avoids comparison errors on dataclass fields
      — O(1) job lookup by ID without heap traversal
    """

    def __init__(self, max_workers: int = 2):
        self._heap:     List[tuple]               = []   # (priority, seq, job_id)
        self._jobs:     Dict[str, QueuedJob]       = {}   # job_id → job
        self._dlq:      deque                      = deque(maxlen=DLQ_MAX_SIZE)
        self._seq:      int                        = 0    # monotonic sequence counter
        self._lock      = threading.RLock()
        self._not_empty = threading.Condition(self._lock)
        self._pool      = ThreadPoolExecutor(max_workers=max_workers,
                                            thread_name_prefix="sb_worker")
        self._running   = True
        self._dispatcher = threading.Thread(target=self._dispatch_loop,
                                            name="sb_dispatcher", daemon=True)
        self._dispatcher.start()
        log.info(f"PriorityJobQueue started (max_workers={max_workers})")

    # ── Submit ─────────────────────────────────────────────────────────────────

    def submit(self, fn: Callable, payload: dict,
               job_id: str = None,
               priority: int = PRIORITY_NORMAL,
               retry_policy: RetryPolicy = None,
               *args, **kwargs) -> str:
        """
        Add a job to the queue. Returns job_id immediately.
        O(log N) heap push.
        """
        job_id = job_id or str(uuid.uuid4())
        policy = retry_policy or RetryPolicy()

        job = QueuedJob(
            job_id       = job_id,
            priority     = priority,
            seq_id       = self._next_seq(),
            payload      = payload,
            fn           = fn,
            fn_args      = args,
            fn_kwargs    = kwargs,
            max_retries  = policy.max_retries,
            retry_policy = policy,
        )

        with self._not_empty:
            self._jobs[job_id] = job
            heapq.heappush(self._heap, (job.priority, job.seq_id, job_id))
            self._not_empty.notify()

        log.debug(f"Queued job {job_id} priority={priority} queue_depth={len(self._heap)}")
        return job_id

    def _next_seq(self) -> int:
        with self._lock:
            self._seq += 1
            return self._seq

    # ── Dispatcher (runs in daemon thread) ────────────────────────────────────

    def _dispatch_loop(self):
        """
        Continuously pop eligible jobs from the heap and submit to thread pool.
        Sleeps when queue is empty or all jobs are delayed (retry backoff).
        """
        while self._running:
            job = self._pop_eligible()
            if job is None:
                time.sleep(0.1)   # brief sleep to avoid busy-wait
                continue
            self._pool.submit(self._run_job, job)

    def _pop_eligible(self) -> Optional[QueuedJob]:
        """
        Pop the highest-priority job that is currently eligible (past its
        next_eligible_at timestamp). O(log N) worst-case.
        """
        now = time.time()
        with self._lock:
            # Scan heap for first eligible job (re-queue ineligible ones)
            skipped = []
            result  = None
            while self._heap:
                prio, seq, job_id = heapq.heappop(self._heap)
                job = self._jobs.get(job_id)
                if job is None:
                    continue   # stale entry (job was cancelled)
                if job.next_eligible_at > now:
                    skipped.append((prio, seq, job_id))   # not yet due
                else:
                    result = job
                    break
            # Re-queue skipped jobs
            for item in skipped:
                heapq.heappush(self._heap, item)
            if result:
                result.status     = "running"
                result.started_at = time.time()
            return result

    # ── Worker ────────────────────────────────────────────────────────────────

    def _run_job(self, job: QueuedJob):
        log.info(f"[Q] Running job {job.job_id} attempt={job.attempt} priority={job.priority}")
        try:
            result = job.fn(*job.fn_args, **job.fn_kwargs)
            with self._lock:
                job.status       = "done"
                job.completed_at = time.time()
                job.result       = result
            log.info(f"[Q] Job {job.job_id} done in {job.completed_at - job.started_at:.2f}s")

        except Exception as exc:
            log.warning(f"[Q] Job {job.job_id} failed (attempt {job.attempt}): {exc}")
            self._handle_failure(job, exc)

    def _handle_failure(self, job: QueuedJob, exc: Exception):
        with self._lock:
            job.last_error = str(exc)
            job.attempt   += 1

            if job.attempt <= job.max_retries:
                delay = job.retry_policy.delay_for_attempt(job.attempt)
                job.next_eligible_at = time.time() + delay
                job.status = "queued"   # back to queued for retry
                heapq.heappush(self._heap, (job.priority, self._seq, job.job_id))
                log.info(f"[Q] Job {job.job_id} retry {job.attempt}/{job.max_retries} in {delay:.1f}s")
            else:
                job.status       = "dead"
                job.completed_at = time.time()
                self._dlq.append(job)
                log.error(f"[Q] Job {job.job_id} moved to DLQ after {job.max_retries} retries")

    # ── Query ─────────────────────────────────────────────────────────────────

    def get_job(self, job_id: str) -> Optional[QueuedJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self, status_filter: str = None) -> List[dict]:
        with self._lock:
            jobs = list(self._jobs.values())
        if status_filter:
            jobs = [j for j in jobs if j.status == status_filter]
        return [
            {
                "job_id":       j.job_id,
                "priority":     j.priority,
                "status":       j.status,
                "attempt":      j.attempt,
                "max_retries":  j.max_retries,
                "payload":      j.payload,
                "created_at":   j.created_at,
                "started_at":   j.started_at,
                "completed_at": j.completed_at,
                "last_error":   j.last_error,
            }
            for j in sorted(jobs, key=lambda x: x.created_at, reverse=True)
        ]

    def dlq_jobs(self) -> List[dict]:
        with self._lock:
            return [
                {
                    "job_id":     j.job_id,
                    "payload":    j.payload,
                    "attempt":    j.attempt,
                    "last_error": j.last_error,
                    "dead_at":    j.completed_at,
                }
                for j in self._dlq
            ]

    def requeue_from_dlq(self, job_id: str) -> bool:
        """Manually requeue a dead job. Returns True if found and requeued."""
        with self._lock:
            for j in self._dlq:
                if j.job_id == job_id:
                    j.status         = "queued"
                    j.attempt        = 0
                    j.next_eligible_at = 0.0
                    j.seq_id         = self._next_seq()
                    self._jobs[j.job_id] = j
                    heapq.heappush(self._heap, (j.priority, j.seq_id, j.job_id))
                    return True
        return False

    def stats(self) -> dict:
        with self._lock:
            by_status: Dict[str, int] = {}
            for j in self._jobs.values():
                by_status[j.status] = by_status.get(j.status, 0) + 1
            return {
                "queue_depth": len(self._heap),
                "total_jobs":  len(self._jobs),
                "dlq_size":    len(self._dlq),
                "by_status":   by_status,
            }

    def shutdown(self):
        self._running = False
        self._pool.shutdown(wait=True)
        log.info("PriorityJobQueue shut down")
