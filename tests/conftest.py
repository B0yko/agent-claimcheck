"""Shared fixtures for the bench test modules.

`offline_recorded_dir` runs the real offline bench once per test session
(retraining classifier-lr, fitting calibrators, timing every offline
detector on all 300 packaged benchmark traces takes a few seconds) and hands
every test module the same recorded directory, read-only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_claimcheck.bench.runner import run_offline
from agent_claimcheck.schema import load_traces


@pytest.fixture(scope="session")
def offline_recorded_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("offline-recorded")
    train = load_traces("bench:train")
    test = load_traces("bench:test")
    run_offline(
        train,
        test,
        out,
        command="agent-claimcheck bench --offline --out results/test",
        date="2026-09-28",
        hardware="test-harness",
        package_version="0.0.0-test",
        dataset_sha256="b" * 64,
    )
    return out
