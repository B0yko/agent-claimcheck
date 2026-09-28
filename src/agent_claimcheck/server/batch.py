"""A bounded worker pool that runs one scoring function over a queue of
trace ids, with live progress, cooperative cancellation and crash-safe
persistence.

A cursor is shared by a fixed pool of worker threads, an `AlreadyRunning`
guard refuses a second concurrent run, cancellation stops new work from
being scheduled but lets any call already in flight finish and be recorded,
and one worker's exception never stops the others. State is written to disk
after every completed item, atomically (tempfile in the same directory,
then `os.replace`), so a reader never sees a half-written file.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

#: A run in one of these statuses is still doing (or about to do) work.
ACTIVE_STATUSES = frozenset({"running", "cancelling"})


class AlreadyRunning(Exception):
    """Raised by `start` when a run is already active."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        super().__init__(f"a run is already active: {run_id}")


class JudgeBatch:
    """Runs `score(trace_id)` over a queue with `concurrency` worker threads.

    `score` is supplied by the caller (the dashboard closes over its own
    `JudgeDetector`, calibrators and gate); this class only owns scheduling,
    progress and persistence. `score` must be thread-safe and must never
    raise for a plain scoring failure (those come back as an
    abstain/verdict from `score` itself); an exception here is treated as
    one worker's own crash and isolated: the item is recorded with
    `verdict: null`, `error: "worker_error"`, and every other item keeps
    running.
    """

    def __init__(self, state_path: str | Path, score: Callable[[str], dict[str, Any]]) -> None:
        self._state_path = Path(state_path)
        self._score = score
        self._lock = threading.RLock()
        self._write_lock = threading.Lock()
        self.run: dict[str, Any] | None = None
        self.thread: threading.Thread | None = None
        self._started_clock = 0.0
        self._cancelled = False

    def start(self, trace_ids: Sequence[str], *, concurrency: int = 4) -> dict[str, Any]:
        """Start a run over `trace_ids`. Raises `AlreadyRunning` if one is active."""
        with self._lock:
            if self.run is not None and (
                self.run["status"] in ACTIVE_STATUSES or (self.thread and self.thread.is_alive())
            ):
                raise AlreadyRunning(self.run["run_id"])
            self._started_clock = time.perf_counter()
            self._cancelled = False
            self.run = {
                "run_id": uuid.uuid4().hex,
                "status": "running",
                "total": len(trace_ids),
                "done": 0,
                "spent_usd": 0.0,
                "active": [],
                "results": [],
            }
            workers = max(1, concurrency)
            self.thread = threading.Thread(
                target=self._execute, args=(list(trace_ids), workers), daemon=True
            )
            initial = copy.deepcopy(self.run)
            self.thread.start()
            return initial

    def snapshot(self, run_id: str | None = None) -> dict[str, Any] | None:
        """A deep copy of the run's current state, or `None` if unknown."""
        with self._lock:
            if self.run is None or (run_id is not None and self.run["run_id"] != run_id):
                return None
            return copy.deepcopy(self.run)

    def cancel(self, run_id: str) -> dict[str, Any] | None:
        """Stop scheduling new items for `run_id`. In-flight items still finish."""
        with self._lock:
            if self.run is None or self.run["run_id"] != run_id:
                return None
            if self.run["status"] in ACTIVE_STATUSES:
                self._cancelled = True
                self.run["status"] = "cancelling"
            return copy.deepcopy(self.run)

    def _persist(self) -> None:
        with self._write_lock:
            with self._lock:
                payload = copy.deepcopy(self.run)
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(
                dir=self._state_path.parent, prefix=".judge-run-", suffix=".json"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, ensure_ascii=False)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp_name, self._state_path)
            except BaseException:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
                raise

    def _persist_quiet(self) -> None:
        try:
            self._persist()
        except OSError:
            with self._lock:
                assert self.run is not None
                self.run["save_error"] = "the run state could not be saved locally"

    def _execute(self, trace_ids: list[str], concurrency: int) -> None:
        cursor = 0

        def worker() -> None:
            nonlocal cursor
            while True:
                with self._lock:
                    assert self.run is not None
                    if self._cancelled or cursor >= len(trace_ids):
                        return
                    trace_id = trace_ids[cursor]
                    cursor += 1
                    self.run["active"].append(trace_id)
                try:
                    item = self._score(trace_id)
                except Exception:
                    item = {
                        "trace_id": trace_id,
                        "verdict": None,
                        "p_success": None,
                        "cost_usd": 0.0,
                        "error": "worker_error",
                    }
                with self._lock:
                    assert self.run is not None
                    self.run["active"].remove(trace_id)
                    self.run["results"].append(item)
                    self.run["done"] += 1
                    spent = float(item.get("cost_usd") or 0.0)
                    self.run["spent_usd"] = round(self.run["spent_usd"] + spent, 6)
                self._persist_quiet()

        try:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(worker) for _ in range(concurrency)]
                for future in futures:
                    future.result()
            with self._lock:
                assert self.run is not None
                self.run["status"] = "cancelled" if self._cancelled else "completed"
        except Exception:
            with self._lock:
                assert self.run is not None
                self.run["status"] = "failed"
        finally:
            with self._lock:
                assert self.run is not None
                self.run["elapsed_ms"] = int((time.perf_counter() - self._started_clock) * 1000)
            self._persist_quiet()
