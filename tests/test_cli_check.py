"""`agent-claimcheck check`: verdict counts, formats, exit codes, and a live
judge run (and budget exhaustion) against a local fake `/chat/completions`
server started inside the test — no real network, no paid call.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from agent_claimcheck.cli import app
from agent_claimcheck.ledger import Price, reservation_usd
from agent_claimcheck.resources import path as resource_path

runner = CliRunner()


def test_example_mixed_has_one_of_each_verdict_per_domain_and_exits_1() -> None:
    result = runner.invoke(app, ["check", "example:mixed", "--format", "jsonl"])
    assert result.exit_code == 1, result.output
    rows = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert len(rows) == 12

    by_domain: dict[str, set[str]] = {}
    for row in rows:
        by_domain.setdefault(row["domain"], set()).add(row["verdict"])
    assert by_domain == {
        "booking": {"verified", "false_success", "unverifiable", "skipped"},
        "crm": {"verified", "false_success", "unverifiable", "skipped"},
        "coding": {"verified", "false_success", "unverifiable", "skipped"},
    }
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    assert counts == {"verified": 3, "false_success": 3, "unverifiable": 3, "skipped": 3}


def test_fail_on_false_success_and_unverifiable() -> None:
    result = runner.invoke(
        app, ["check", "example:mixed", "--fail-on", "false_success,unverifiable"]
    )
    assert result.exit_code == 1


def test_fail_on_a_verdict_absent_from_the_results_exits_0() -> None:
    # example:browser has no rule pack and the default classifier-lr
    # abstains out-of-distribution for it, so cascade-offline never gates
    # anything `verified` or `false_success` there.
    result = runner.invoke(app, ["check", "example:browser", "--fail-on", "verified"])
    assert result.exit_code == 0, result.output


def test_exits_2_on_bad_fail_on_value() -> None:
    result = runner.invoke(app, ["check", "example:mixed", "--fail-on", "bogus"])
    assert result.exit_code == 2


def test_exits_2_on_missing_input_file() -> None:
    result = runner.invoke(app, ["check", "/no/such/file.jsonl"])
    assert result.exit_code == 2


def test_exits_2_on_unknown_detector() -> None:
    result = runner.invoke(app, ["check", "example:mixed", "--detector", "not-a-detector"])
    assert result.exit_code == 2


def test_exits_2_on_bad_format() -> None:
    result = runner.invoke(app, ["check", "example:mixed", "--format", "yaml"])
    assert result.exit_code == 2


def test_json_output_validates_against_the_result_schema() -> None:
    result = runner.invoke(app, ["check", "example:mixed", "--format", "json"])
    rows = json.loads(result.stdout)
    assert len(rows) == 12
    _assert_all_valid(rows)


def test_jsonl_output_validates_against_the_result_schema() -> None:
    result = runner.invoke(app, ["check", "example:mixed", "--format", "jsonl"])
    rows = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    _assert_all_valid(rows)


def _assert_all_valid(rows: list[dict[str, Any]]) -> None:
    from jsonschema import Draft202012Validator

    schema = json.loads(resource_path("schemas/claimcheck-result-v1.json").read_text())
    validator = Draft202012Validator(schema)
    for row in rows:
        errors = list(validator.iter_errors(row))
        assert not errors, errors


def test_example_browser_with_classifier_is_only_unverifiable_or_skipped() -> None:
    result = runner.invoke(
        app, ["check", "example:browser", "--detector", "classifier", "--format", "jsonl"]
    )
    rows = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert {r["verdict"] for r in rows} <= {"unverifiable", "skipped"}
    assert len(rows) == 24


def test_out_writes_jsonl_results(tmp_path: Path) -> None:
    out_path = tmp_path / "results.jsonl"
    result = runner.invoke(app, ["check", "example:mixed", "--out", str(out_path)])
    assert result.exit_code == 1
    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 12
    for line in lines:
        json.loads(line)  # each line is valid JSON on its own


def test_examples_probes_file_changes_a_specific_trace_verdict() -> None:
    # Without --probes: booking-01's receipt has no matching state_probe, so
    # `rules` alone gates it `unverifiable` (receipt_only, 0.70).
    without = runner.invoke(
        app, ["check", "example:mixed", "--detector", "rules", "--format", "jsonl"]
    )
    rows = {json.loads(line)["trace_id"]: json.loads(line) for line in without.stdout.splitlines()}
    assert rows["booking-01"]["verdict"] == "unverifiable"

    # examples/probes.jsonl supplies exactly that probe for booking-01;
    # merging it in resolves the claim to probe_supported (0.97, verified).
    probes = resource_path("examples/probes.jsonl")
    with_probes = runner.invoke(
        app,
        [
            "check",
            "example:mixed",
            "--probes",
            str(probes),
            "--detector",
            "rules",
            "--format",
            "jsonl",
        ],
    )
    rows2 = {
        json.loads(line)["trace_id"]: json.loads(line) for line in with_probes.stdout.splitlines()
    }
    assert rows2["booking-01"]["verdict"] == "verified"
    assert rows2["booking-01"]["p_success_raw"] == pytest.approx(0.97)


def test_note_printed_for_builtin_calibrators_on_non_bench_input() -> None:
    result = runner.invoke(app, ["check", "example:mixed"])
    assert "built-in calibrators were fitted on the synthetic benchmark" in result.stderr


def test_no_note_when_bench_train_is_the_input() -> None:
    result = runner.invoke(app, ["check", "bench:train", "--format", "jsonl"])
    assert "built-in calibrators were fitted" not in result.stderr


@pytest.mark.parametrize("detector", ["trust-agent", "any-error"])
def test_no_note_when_no_builtin_calibrator_matches_the_detector(detector: str) -> None:
    # The built-in calibrator set only has `rules` and `classifier-lr` keys,
    # so baseline detectors never get calibrated even though
    # `checker.calibrators_builtin` is true: the note would be misleading
    # here.
    result = runner.invoke(
        app, ["check", "example:mixed", "--detector", detector, "--format", "jsonl"]
    )
    rows = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert rows and not any(row["calibrated"] for row in rows)
    assert "built-in calibrators were fitted" not in result.stderr


def test_calibration_flag_overrides_the_builtin_set(tmp_path: Path) -> None:
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "version": 1,
                "fitted_on": "custom",
                "base_rate": 0.6,
                "calibrators": {
                    "rules": {
                        "method": "platt",
                        "a": 0.05,
                        "b": -1.0,
                        "clip": 1e-6,
                        "n": 10,
                        "n_pos": 5,
                        "fitted_on": "custom",
                        "detector": "rules",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        [
            "check",
            "example:mixed",
            "--detector",
            "rules",
            "--calibration",
            str(calibration_path),
            "--format",
            "jsonl",
        ],
    )
    assert "built-in calibrators were fitted" not in result.stderr
    rows = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    booking_01 = next(r for r in rows if r["trace_id"] == "booking-01")
    # booking-01 has no probe in the base example set, so `rules` alone
    # gates it receipt_only (raw 0.70); the custom calibrator
    # (a=0.05, b=-1.0) maps that to ~0.277.
    assert booking_01["p_success_raw"] == pytest.approx(0.70)
    assert booking_01["p_success"] == pytest.approx(0.27735193561025356)


# ------------------------------------------------------- invalid input lines --


def _mixed_validity_file(tmp_path: Path) -> Path:
    valid = tmp_path / "valid.jsonl"
    _write_labelled_traces(valid, 2)
    good_lines = valid.read_text(encoding="utf-8").splitlines()
    path = tmp_path / "mixed.jsonl"
    path.write_text("\n".join([good_lines[0], "{not json", good_lines[1]]) + "\n", encoding="utf-8")
    return path


def test_invalid_lines_are_skipped_but_exit_2_after_the_valid_results(tmp_path: Path) -> None:
    path = _mixed_validity_file(tmp_path)
    result = runner.invoke(app, ["check", str(path), "--detector", "rules", "--format", "jsonl"])
    assert result.exit_code == 2, result.output
    rows = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert [r["trace_id"] for r in rows] == ["t0", "t1"]
    assert "line 2:" in result.stderr
    assert "1 invalid trace line(s) skipped" in result.stderr


def test_invalid_lines_dominate_fail_on(tmp_path: Path) -> None:
    path = _mixed_validity_file(tmp_path)
    # Every verdict fails the run, yet a validation error is still exit 2, not 1.
    result = runner.invoke(
        app,
        [
            "check",
            str(path),
            "--detector",
            "rules",
            "--fail-on",
            "verified,false_success,unverifiable,skipped",
        ],
    )
    assert result.exit_code == 2, result.output


def test_out_still_written_for_valid_lines_when_some_are_invalid(tmp_path: Path) -> None:
    path = _mixed_validity_file(tmp_path)
    out_path = tmp_path / "results.jsonl"
    result = runner.invoke(app, ["check", str(path), "--detector", "rules", "--out", str(out_path)])
    assert result.exit_code == 2
    assert len(out_path.read_text(encoding="utf-8").strip().splitlines()) == 2


def test_strict_aborts_on_the_first_invalid_line_without_results(tmp_path: Path) -> None:
    path = _mixed_validity_file(tmp_path)
    out_path = tmp_path / "results.jsonl"
    result = runner.invoke(
        app,
        [
            "check",
            str(path),
            "--detector",
            "rules",
            "--strict",
            "--format",
            "jsonl",
            "--out",
            str(out_path),
        ],
    )
    assert result.exit_code == 2, result.output
    assert result.stdout.strip() == ""
    assert "line 2:" in result.stderr
    assert not out_path.exists()


def test_strict_on_a_clean_file_behaves_like_the_default() -> None:
    result = runner.invoke(app, ["check", "example:mixed", "--strict", "--format", "jsonl"])
    assert result.exit_code == 1  # the false_success traces, as without --strict
    assert len(result.stdout.splitlines()) == 12


def test_an_empty_input_file_exits_2(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    result = runner.invoke(app, ["check", str(path)])
    assert result.exit_code == 2
    assert "no valid traces" in result.stderr


def test_an_input_with_only_invalid_lines_exits_2(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("{not json\n[]\n", encoding="utf-8")
    result = runner.invoke(app, ["check", str(path), "--fail-on", "skipped"])
    assert result.exit_code == 2
    assert "no valid traces" in result.stderr
    assert "line 1:" in result.stderr


# --------------------------------------------------------------- live judge --

_VALID_CONTENT = json.dumps(
    {
        "p_success": 0.9,
        "failure_kind": "none",
        "evidence_steps": [1],
        "rationale": "the receipt confirms the claim",
    }
)


def _fake_server(handler_cls: type[BaseHTTPRequestHandler]) -> tuple[ThreadingHTTPServer, int]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, port


def _write_labelled_traces(path: Path, n: int) -> None:
    # The judge request carries no trace or task id, so identical traces
    # share one cache key: a trace scored after another's live call has
    # been cached would be a cache hit, not a call. A per-trace instruction
    # keeps every request distinct, whatever the thread timing.
    lines = []
    for i in range(n):
        trace = {
            "schema": "agent-trace/v1",
            "trace_id": f"t{i}",
            "source": "test/0.0.1",
            "task": {"id": f"t{i}-task", "domain": "booking", "instruction": f"book slot {i}"},
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
                    "ok": True,
                    "output": {"event_id": "e1"},
                },
            ],
            "final_claim": {
                "text": "Booked it.",
                "claims": [{"type": "booked", "subject": {}}],
            },
        }
        lines.append(json.dumps(trace))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_check_detector_judge_against_local_fake_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            payload = json.dumps(
                {
                    "choices": [{"message": {"content": _VALID_CONTENT}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.001},
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            pass

    server, port = _fake_server(Handler)
    try:
        traces_path = tmp_path / "traces.jsonl"
        _write_labelled_traces(traces_path, 2)
        config_path = tmp_path / "claimcheck.toml"
        config_path.write_text(
            "[judge]\n"
            f'base_url = "http://127.0.0.1:{port}/v1"\n'
            'model = "test/fake-model"\n'
            "price_in_per_m = 1.0\n"
            "price_out_per_m = 1.0\n",
            encoding="utf-8",
        )
        ledger_path = tmp_path / "ledger.jsonl"
        monkeypatch.setenv("CLAIMCHECK_API_KEY", "test-key")
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
        monkeypatch.setenv("CLAIMCHECK_LEDGER", str(ledger_path))

        result = runner.invoke(
            app,
            [
                "check",
                str(traces_path),
                "--detector",
                "judge",
                "--config",
                str(config_path),
                "--format",
                "jsonl",
                "--fail-on",
                "unverifiable",
                "--no-cache",  # both traces render the same request
            ],
        )
        assert result.exit_code == 0, result.output
        rows = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
        assert len(rows) == 2
        assert all(r["verdict"] == "verified" for r in rows)
        assert all(r["p_success_raw"] == pytest.approx(0.9) for r in rows)
        # The built-in calibrator set has no `judge` key, so nothing here
        # was actually calibrated and the built-in-calibrators note must
        # not print.
        assert all(not r["calibrated"] for r in rows)
        assert "built-in calibrators were fitted" not in result.stderr

        ledger_lines = ledger_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(ledger_lines) == 2
        for line in ledger_lines:
            entry = json.loads(line)
            assert entry["status"] == "ok"
            assert "Booked it" not in line
    finally:
        server.shutdown()


def test_check_detector_judge_budget_exhaustion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    call_count = {"n": 0}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            call_count["n"] += 1
            payload = json.dumps(
                {
                    "choices": [{"message": {"content": _VALID_CONTENT}}],
                    # Cost equals the reservation exactly (computed below),
                    # so spend tracks reservations precisely.
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": RESERVATION},
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            pass

    # price_in_per_m = 0 makes the reservation independent of request size:
    # purely max_tokens (400, the default) * price_out_per_m / 1e6 * 1.10.
    price_out_per_m = 1_000_000.0
    RESERVATION = reservation_usd(
        "", 400, Price(price_in_per_m=0.0, price_out_per_m=price_out_per_m)
    )
    max_usd = RESERVATION * 1.5  # exactly one call's worth of headroom

    server, port = _fake_server(Handler)
    try:
        traces_path = tmp_path / "traces.jsonl"
        _write_labelled_traces(traces_path, 3)
        config_path = tmp_path / "claimcheck.toml"
        config_path.write_text(
            "[judge]\n"
            f'base_url = "http://127.0.0.1:{port}/v1"\n'
            'model = "test/fake-model"\n'
            "price_in_per_m = 0.0\n"
            f"price_out_per_m = {price_out_per_m}\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("CLAIMCHECK_API_KEY", "test-key")
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
        monkeypatch.setenv("CLAIMCHECK_LEDGER", str(tmp_path / "ledger.jsonl"))

        result = runner.invoke(
            app,
            [
                "check",
                str(traces_path),
                "--detector",
                "judge",
                "--config",
                str(config_path),
                "--format",
                "jsonl",
                "--max-usd",
                str(max_usd),
                "--fail-on",
                "unverifiable",
                # The three traces render identical requests; without this a
                # later trace could be served from the first one's cache entry.
                "--no-cache",
            ],
        )
        rows = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
        assert len(rows) == 3
        assert not any(r["cached"] for r in rows)
        verdicts = [r["verdict"] for r in rows]
        assert verdicts.count("verified") == 1
        assert verdicts.count("unverifiable") == 2
        exhausted = [r for r in rows if r["verdict"] == "unverifiable"]
        assert all(r["abstain_reason"] == "budget_exhausted" for r in exhausted)
        assert call_count["n"] == 1  # the other two never reached the server
        assert result.exit_code == 1  # --fail-on unverifiable
    finally:
        server.shutdown()


def test_check_judge_reuses_the_cache_unless_no_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"n": 0}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            calls["n"] += 1
            payload = json.dumps(
                {
                    "choices": [{"message": {"content": _VALID_CONTENT}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.0001},
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            pass

    server, port = _fake_server(Handler)
    try:
        traces_path = tmp_path / "traces.jsonl"
        _write_labelled_traces(traces_path, 1)
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
        monkeypatch.setenv("CLAIMCHECK_LEDGER", str(tmp_path / "ledger.jsonl"))
        args = ["check", str(traces_path), "--detector", "judge", "--config", str(config_path)]
        args += ["--format", "jsonl"]

        first = json.loads(runner.invoke(app, args).stdout.splitlines()[0])
        second = json.loads(runner.invoke(app, args).stdout.splitlines()[0])
        assert calls["n"] == 1
        assert first["cached"] is False
        assert second["cached"] is True
        assert second["cost_usd"] == 0.0

        third = json.loads(runner.invoke(app, [*args, "--no-cache"]).stdout.splitlines()[0])
        assert calls["n"] == 2
        assert third["cached"] is False
    finally:
        server.shutdown()


# ------------------------------------------------------------ judge pricing --


class _CountingHandler(BaseHTTPRequestHandler):
    """Serves `/models` (ids in `listed`, $1/M both ways) and counts completions."""

    listed: tuple[str, ...] = ()
    usage: dict[str, Any] = {"prompt_tokens": 100_000, "completion_tokens": 0}
    posts = 0

    def _send(self, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path.endswith("/endpoints"):
            self._send({"data": {"endpoints": []}})
            return
        listing = [
            {
                "id": model,
                "pricing": {"prompt": "0.000001", "completion": "0.000001"},
                "supported_parameters": [],
            }
            for model in self.listed
        ]
        self._send({"data": listing})

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        type(self).posts += 1
        self._send({"choices": [{"message": {"content": _VALID_CONTENT}}], "usage": self.usage})

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass


def _pricing_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    listed: tuple[str, ...],
    config_prices: bool,
) -> tuple[ThreadingHTTPServer, type[_CountingHandler], list[str]]:
    handler = type("Handler", (_CountingHandler,), {"listed": listed, "posts": 0})
    server, port = _fake_server(handler)
    traces_path = tmp_path / "traces.jsonl"
    _write_labelled_traces(traces_path, 1)
    config_path = tmp_path / "claimcheck.toml"
    config_path.write_text(
        "[judge]\n"
        f'base_url = "http://127.0.0.1:{port}/v1"\n'
        'model = "test/fake-model"\n'
        + ("price_in_per_m = 1.0\nprice_out_per_m = 1.0\n" if config_prices else ""),
        encoding="utf-8",
    )
    monkeypatch.setenv("CLAIMCHECK_API_KEY", "test-key")
    monkeypatch.setenv("CLAIMCHECK_LEDGER", str(tmp_path / "ledger.jsonl"))
    args = ["check", str(traces_path), "--config", str(config_path), "--format", "jsonl"]
    args += ["--fail-on", "unverifiable", "--no-cache"]
    return server, handler, args


@pytest.mark.parametrize("detector", ["judge", "cascade"])
def test_check_refuses_up_front_when_the_judge_has_no_known_price(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, detector: str
) -> None:
    server, handler, args = _pricing_setup(tmp_path, monkeypatch, listed=(), config_prices=False)
    try:
        result = runner.invoke(app, [*args, "--detector", detector])
        assert result.exit_code == 2, result.output
        assert result.stdout.strip() == ""  # no per-trace abstentions were printed
        assert len(result.stderr.strip().splitlines()) == 1
        assert result.stderr.startswith("error: no price known for judge model 'test/fake-model'")
        assert "--price-in and --price-out" in result.stderr
        assert "[judge]" in result.stderr  # not swallowed as rich markup
        assert handler.posts == 0
    finally:
        server.shutdown()


def test_check_price_flags_supply_a_missing_price(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, _handler, args = _pricing_setup(tmp_path, monkeypatch, listed=(), config_prices=False)
    try:
        result = runner.invoke(
            app, [*args, "--detector", "judge", "--price-in", "1", "--price-out", "1"]
        )
        assert result.exit_code == 0, result.output
        [row] = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
        assert row["verdict"] == "verified"
        assert row["abstain"] is False
        assert row["cost_usd"] == pytest.approx(0.1)  # 100k prompt tokens at $1/M
    finally:
        server.shutdown()


def test_check_price_flags_beat_the_config_price(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, _handler, args = _pricing_setup(tmp_path, monkeypatch, listed=(), config_prices=True)
    try:
        result = runner.invoke(
            app, [*args, "--detector", "judge", "--price-in", "3", "--price-out", "3"]
        )
        assert result.exit_code == 0, result.output
        [row] = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
        assert row["cost_usd"] == pytest.approx(0.3)  # the flag's $3/M, not the config's $1/M
    finally:
        server.shutdown()


def test_check_uses_the_listing_price_when_neither_flags_nor_config_set_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, _handler, args = _pricing_setup(
        tmp_path, monkeypatch, listed=("test/fake-model",), config_prices=False
    )
    try:
        result = runner.invoke(app, [*args, "--detector", "judge"])
        assert result.exit_code == 0, result.output
        [row] = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
        assert row["cost_usd"] == pytest.approx(0.1)
    finally:
        server.shutdown()


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        (["--price-in", "1"], "must be given together"),
        (["--price-out", "1"], "must be given together"),
        (["--price-in", "-1", "--price-out", "1"], "must not be negative"),
    ],
)
def test_check_price_flags_are_both_or_neither(flags: list[str], message: str) -> None:
    result = runner.invoke(app, ["check", "example:mixed", *flags])
    assert result.exit_code == 2
    assert message in result.stderr


def test_check_price_flags_are_ignored_by_detectors_without_a_judge() -> None:
    result = runner.invoke(
        app,
        ["check", "example:mixed", "--price-in", "1", "--price-out", "1", "--format", "jsonl"],
    )
    assert result.exit_code == 1  # false_success verdicts, not an input error
