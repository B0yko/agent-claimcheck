"""Tests for the `agent-claimcheck train` command."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from agent_claimcheck.cli import app

runner = CliRunner()


def _labelled_trace(trace_id: str, domain: str, outcome: str) -> dict[str, object]:
    ok = outcome == "success"
    return {
        "schema": "agent-trace/v1",
        "trace_id": trace_id,
        "source": "test/0.0.1",
        "task": {"id": f"{trace_id}-task", "domain": domain, "instruction": "book a slot"},
        "steps": [
            {
                "i": 0,
                "ts": "2026-01-01T00:00:00+00:00",
                "kind": "tool_call",
                "role": "agent",
                "name": "calendar.create_event",
                "args": {"start": "2026-01-01T01:00:00+00:00"},
            },
            {
                "i": 1,
                "ts": "2026-01-01T00:00:01+00:00",
                "kind": "tool_result",
                "role": "tool",
                "name": "calendar.create_event",
                "ok": ok,
                "output": {"event_id": "e1", "start": "2026-01-01T01:00:00+00:00"} if ok else {},
                "error": None if ok else "409: conflict",
            },
        ],
        "final_claim": {
            "text": "Booked for 2026-01-01T01:00:00+00:00.",
            "claims": [
                {
                    "type": "booked",
                    "subject": {"start": "2026-01-01T01:00:00+00:00"},
                }
            ],
        },
        "ground_truth": {"outcome": outcome, "checked_by": "state_probe", "details": {}},
    }


def _unlabelled_trace(trace_id: str) -> dict[str, object]:
    t = _labelled_trace(trace_id, "booking", "success")
    t["ground_truth"] = {"outcome": "unknown", "checked_by": "none", "details": {}}
    return t


def _no_claim_trace(trace_id: str) -> dict[str, object]:
    t = _labelled_trace(trace_id, "booking", "success")
    t["final_claim"] = {"text": "I couldn't book it.", "claims": []}
    return t


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_train_writes_lr_and_calibration_json(tmp_path: Path) -> None:
    rows = [_labelled_trace(f"g{i}", "booking", "success") for i in range(8)] + [
        _labelled_trace(f"f{i}", "booking", "failure") for i in range(8)
    ]
    input_path = tmp_path / "labelled.jsonl"
    _write_jsonl(input_path, rows)
    out_dir = tmp_path / "out"

    result = runner.invoke(app, ["train", str(input_path), "--out", str(out_dir)])
    assert result.exit_code == 0, result.output

    artifact = json.loads((out_dir / "lr-v1.json").read_text(encoding="utf-8"))
    assert artifact["format"] == "claimcheck-lr/v1"
    assert artifact["training_domains"] == ["booking"]
    assert artifact["n_train"] == 16

    calibration = json.loads((out_dir / "calibration.json").read_text(encoding="utf-8"))
    assert set(calibration["calibrators"]) == {"classifier-lr"}


def test_train_calibrate_rules_adds_a_rules_calibrator(tmp_path: Path) -> None:
    rows = [_labelled_trace(f"g{i}", "booking", "success") for i in range(8)] + [
        _labelled_trace(f"f{i}", "booking", "failure") for i in range(8)
    ]
    input_path = tmp_path / "labelled.jsonl"
    _write_jsonl(input_path, rows)
    out_dir = tmp_path / "out"

    result = runner.invoke(
        app, ["train", str(input_path), "--out", str(out_dir), "--calibrate", "rules"]
    )
    assert result.exit_code == 0, result.output
    calibration = json.loads((out_dir / "calibration.json").read_text(encoding="utf-8"))
    assert "rules" in calibration["calibrators"]


def test_train_rejects_unsupported_calibrate_key(tmp_path: Path) -> None:
    rows = [_labelled_trace(f"g{i}", "booking", "success") for i in range(4)]
    input_path = tmp_path / "labelled.jsonl"
    _write_jsonl(input_path, rows)

    result = runner.invoke(
        app, ["train", str(input_path), "--out", str(tmp_path / "out"), "--calibrate", "bogus"]
    )
    assert result.exit_code == 2
    assert "bogus" in result.output


def test_train_calibrate_judge_runs_the_configured_judge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = json.dumps(
        {
            "p_success": 0.8,
            "failure_kind": "none",
            "evidence_steps": [1],
            "rationale": "looks fine",
        }
    )

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            payload = json.dumps(
                {
                    "choices": [{"message": {"content": content}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 5, "cost": 0.0001},
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        rows = [_labelled_trace(f"g{i}", "booking", "success") for i in range(8)] + [
            _labelled_trace(f"f{i}", "booking", "failure") for i in range(8)
        ]
        input_path = tmp_path / "labelled.jsonl"
        _write_jsonl(input_path, rows)
        out_dir = tmp_path / "out"

        config_path = tmp_path / "claimcheck.toml"
        config_path.write_text(
            "[judge]\n"
            f'base_url = "http://127.0.0.1:{port}/v1"\n'
            'model = "test/fake-model"\n'
            "price_in_per_m = 1.0\n"
            "price_out_per_m = 1.0\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("CLAIMCHECK_API_KEY", "test-key")
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
        monkeypatch.setenv("CLAIMCHECK_LEDGER", str(tmp_path / "ledger.jsonl"))

        result = runner.invoke(
            app,
            [
                "train",
                str(input_path),
                "--out",
                str(out_dir),
                "--calibrate",
                "judge",
                "--config",
                str(config_path),
            ],
        )
        assert result.exit_code == 0, result.output

        calibration = json.loads((out_dir / "calibration.json").read_text(encoding="utf-8"))
        assert "judge:test/fake-model" in calibration["calibrators"]

        ledger_lines = (tmp_path / "ledger.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(ledger_lines) == 16
    finally:
        server.shutdown()


def test_train_skips_unlabelled_and_no_success_claim_traces(tmp_path: Path) -> None:
    rows = (
        [_labelled_trace(f"g{i}", "booking", "success") for i in range(5)]
        + [_labelled_trace(f"f{i}", "booking", "failure") for i in range(5)]
        + [_unlabelled_trace("u1"), _unlabelled_trace("u2")]
        + [_no_claim_trace("n1")]
    )
    input_path = tmp_path / "labelled.jsonl"
    _write_jsonl(input_path, rows)
    out_dir = tmp_path / "out"

    result = runner.invoke(app, ["train", str(input_path), "--out", str(out_dir)])
    assert result.exit_code == 0, result.output
    assert "skipped 2 unlabelled, 1 without a success claim" in result.output
    artifact = json.loads((out_dir / "lr-v1.json").read_text(encoding="utf-8"))
    assert artifact["n_train"] == 10
