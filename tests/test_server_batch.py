"""`JudgeBatch`: scheduling, cancellation and crash-safe persistence."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest

from agent_claimcheck.server.batch import AlreadyRunning, JudgeBatch


def test_cancel_stops_scheduling_and_keeps_inflight_results(tmp_path: Path) -> None:
    entered = threading.Event()
    release = threading.Event()
    lock = threading.Lock()
    calls: list[str] = []

    def score(trace_id: str) -> dict[str, Any]:
        with lock:
            calls.append(trace_id)
            if len(calls) == 2:
                entered.set()
        assert release.wait(timeout=3)
        return {"trace_id": trace_id, "verdict": "verified", "p_success": 0.9, "cost_usd": 0.01}

    batch = JudgeBatch(tmp_path / "run.json", score)
    trace_ids = [f"t{i}" for i in range(5)]
    initial = batch.start(trace_ids, concurrency=2)
    try:
        assert entered.wait(timeout=3)
        with pytest.raises(AlreadyRunning) as exc_info:
            batch.start(trace_ids, concurrency=2)
        assert exc_info.value.run_id == initial["run_id"]

        cancelled = batch.cancel(initial["run_id"])
        assert cancelled is not None
        assert cancelled["status"] == "cancelling"
        assert len(cancelled["active"]) == 2
    finally:
        release.set()
        assert batch.thread is not None
        batch.thread.join(timeout=5)

    snapshot = batch.snapshot(initial["run_id"])
    assert snapshot is not None
    assert snapshot["status"] == "cancelled"
    assert snapshot["done"] == 2
    assert snapshot["active"] == []
    assert len(calls) == 2

    saved = json.loads((tmp_path / "run.json").read_text())
    assert saved["run_id"] == initial["run_id"]
    assert saved["status"] == "cancelled"
    assert not list(tmp_path.glob(".judge-run-*"))


def test_worker_error_is_isolated_and_the_run_still_finishes(tmp_path: Path) -> None:
    def score(trace_id: str) -> dict[str, Any]:
        if trace_id == "bad":
            raise RuntimeError("boom, and this text must never reach the snapshot")
        return {"trace_id": trace_id, "verdict": "verified", "p_success": 0.9, "cost_usd": 0.0}

    batch = JudgeBatch(tmp_path / "run.json", score)
    batch.start(["good-1", "bad", "good-2"], concurrency=1)
    assert batch.thread is not None
    batch.thread.join(timeout=5)

    snapshot = batch.snapshot()
    assert snapshot is not None
    assert snapshot["status"] == "completed"
    assert snapshot["done"] == 3
    by_id = {item["trace_id"]: item for item in snapshot["results"]}
    assert by_id["bad"]["verdict"] is None
    assert by_id["bad"]["error"] == "worker_error"
    assert by_id["good-1"]["verdict"] == "verified"
    assert "boom" not in json.dumps(snapshot)


def test_already_running_reports_the_active_run_id(tmp_path: Path) -> None:
    release = threading.Event()

    def score(trace_id: str) -> dict[str, Any]:
        assert release.wait(timeout=3)
        return {"trace_id": trace_id, "verdict": "verified", "p_success": 0.9, "cost_usd": 0.0}

    batch = JudgeBatch(tmp_path / "run.json", score)
    first = batch.start(["a", "b"], concurrency=1)
    try:
        with pytest.raises(AlreadyRunning) as exc_info:
            batch.start(["a", "b"], concurrency=1)
        assert exc_info.value.run_id == first["run_id"]
    finally:
        release.set()
        assert batch.thread is not None
        batch.thread.join(timeout=5)


def test_snapshot_and_cancel_return_none_for_an_unknown_run(tmp_path: Path) -> None:
    batch = JudgeBatch(tmp_path / "run.json", lambda trace_id: {"trace_id": trace_id})
    assert batch.snapshot("nope") is None
    assert batch.cancel("nope") is None


def test_persist_writes_atomically_no_partial_file_left_behind(tmp_path: Path) -> None:
    def score(trace_id: str) -> dict[str, Any]:
        return {"trace_id": trace_id, "verdict": "verified", "p_success": 0.9, "cost_usd": 0.0}

    state_path = tmp_path / "nested" / "run.json"
    batch = JudgeBatch(state_path, score)
    batch.start(["a", "b", "c"], concurrency=2)
    assert batch.thread is not None
    batch.thread.join(timeout=5)

    assert state_path.exists()
    saved = json.loads(state_path.read_text())
    assert saved["done"] == 3
    assert not list(state_path.parent.glob(".judge-run-*"))
