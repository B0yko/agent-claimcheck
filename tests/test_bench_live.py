"""`bench --live`: dry-run reservations and the real judged run.

A fake local OpenRouter-shaped server (`/models`, `/models/{model}/endpoints`,
`/chat/completions`) runs in a thread for every test here; nothing in this
file makes a real network call.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from agent_claimcheck.bench.live import (
    BudgetPreflightError,
    dry_run_plan,
    format_spend_summary,
    run_live,
)
from agent_claimcheck.bench.report import build_report, verify_judge_requests
from agent_claimcheck.cli import app
from agent_claimcheck.judge.render import render_request
from agent_claimcheck.ledger import Price, reservation_usd
from agent_claimcheck.schema import load_traces

runner = CliRunner()

MODEL_A = "test/judge-a"
MODEL_B = "test/judge-b"

MODELS_PAYLOAD = {
    "data": [
        {
            "id": MODEL_A,
            "pricing": {"prompt": "0.0000001", "completion": "0.0000002"},
            "supported_parameters": ["response_format"],
        },
        {
            "id": MODEL_B,
            "pricing": {"prompt": "0.0000003", "completion": "0.0000004"},
            "supported_parameters": ["response_format"],
        },
    ]
}

ENDPOINTS_PAYLOAD = {
    MODEL_A: {
        "data": {"endpoints": [{"pricing": {"prompt": "0.00000015", "completion": "0.0000005"}}]}
    },
    MODEL_B: {
        "data": {"endpoints": [{"pricing": {"prompt": "0.0000004", "completion": "0.0000006"}}]}
    },
}


def _judgment(p_success: float) -> str:
    return json.dumps(
        {
            "p_success": p_success,
            "failure_kind": "none",
            "evidence_steps": [],
            "rationale": "fake judge answer",
        }
    )


class _FakeORHandler(BaseHTTPRequestHandler):
    """Routes by path suffix so it works whether it is hit directly (a plain
    `http://127.0.0.1:PORT/v1` base URL) or through a transport that keeps
    the caller's own base URL (see `_proxy_transport`).
    """

    call_count = 0
    post_paths: list[str] = []
    get_paths: list[str] = []
    api_keys_seen: list[str] = []

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - http.server's naming convention
        cls = type(self)
        cls.get_paths.append(self.path)
        if self.path.endswith("/models"):
            self._send_json(200, MODELS_PAYLOAD)
            return
        if "/models/" in self.path and self.path.endswith("/endpoints"):
            model = self.path.split("/models/", 1)[1].rsplit("/endpoints", 1)[0]
            self._send_json(200, ENDPOINTS_PAYLOAD.get(model, {"data": {"endpoints": []}}))
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        cls = type(self)
        cls.post_paths.append(self.path)
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        auth = self.headers.get("Authorization", "")
        cls.api_keys_seen.append(auth)
        cls.call_count += 1
        content = _judgment(0.9 if cls.call_count % 2 else 0.15)
        self._send_json(
            200,
            {
                "choices": [{"message": {"content": content}}],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 8,
                    "cost": 0.00015,
                },
            },
        )

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass


@pytest.fixture
def fake_server() -> Any:
    _FakeORHandler.call_count = 0
    _FakeORHandler.post_paths = []
    _FakeORHandler.get_paths = []
    _FakeORHandler.api_keys_seen = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeORHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield port
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _proxy_transport(port: int) -> httpx.MockTransport:
    """A transport that answers every request from the real local fake
    server above while letting the caller keep `base_url` pointed at
    `https://openrouter.ai/api/v1` (needed so a later replay, which always
    assumes that host, recomputes the same request bodies).
    """
    upstream = httpx.Client(base_url=f"http://127.0.0.1:{port}")

    def handler(request: httpx.Request) -> httpx.Response:
        response = upstream.request(request.method, request.url.path, content=request.content)
        return httpx.Response(response.status_code, content=response.content)

    return httpx.MockTransport(handler)


def test_dry_run_totals_equal_sum_of_per_request_reservations(fake_server: int) -> None:
    port = fake_server
    train = load_traces("bench:train")
    test = load_traces("bench:test")
    base_url = f"http://127.0.0.1:{port}/v1"

    rows, grand_total = dry_run_plan(
        train,
        test,
        judges=[MODEL_A, MODEL_B],
        ablation_judge=MODEL_A,
        ablation_prompt="claim-by-claim",
        limit=3,
        base_url=base_url,
        api_key="k",
        price_override=None,
    )

    assert [r.n_calls for r in rows] == [6, 6, 6]  # 3 train + 3 test each
    assert _FakeORHandler.post_paths == []  # a dry run sends no chat requests

    # Recompute independently, straight from `reservation_usd`/`render_request`,
    # the same formula `plan_reservations` uses internally, over the same
    # (model, prompt) combinations `build_runs` would produce.
    from agent_claimcheck.bench.live import build_runs
    from agent_claimcheck.claims import ClaimExtractor
    from agent_claimcheck.ledger import PriceBook
    from agent_claimcheck.redact import detector_view, resolve_claims
    from agent_claimcheck.rules.engine import builtin_packs

    extractor = ClaimExtractor(list(builtin_packs().values()))
    views = [resolve_claims(detector_view(t), extractor) for t in (*train[:3], *test[:3])]

    with httpx.Client() as client:
        price_book = PriceBook(base_url=base_url, api_key="k", client=client)
        runs = build_runs([MODEL_A, MODEL_B], MODEL_A, "claim-by-claim", price_book, base_url)
        expected_total = 0.0
        for run in runs:
            max_price = price_book.max_price(run.model)
            for view in views:
                body = render_request(view, run.prompt, run.spec)
                request_json = json.dumps(
                    body, sort_keys=True, ensure_ascii=False, separators=(",", ":")
                )
                expected_total += reservation_usd(request_json, run.spec.max_tokens, max_price)

    assert grand_total == pytest.approx(expected_total)
    assert grand_total == pytest.approx(sum(r.total_usd for r in rows))


def test_live_mini_run_writes_every_file_and_leaks_no_key(fake_server: int, tmp_path: Path) -> None:
    port = fake_server
    train = load_traces("bench:train")
    test = load_traces("bench:test")
    out_dir = tmp_path / "run"
    secret = "sk-test-super-secret-key"

    summary = run_live(
        train,
        test,
        out_dir,
        judges=[MODEL_A, MODEL_B],
        ablation_judge=MODEL_A,
        ablation_prompt="claim-by-claim",
        run_name="t1",
        max_usd=5.0,
        concurrency=2,
        limit=3,
        allow_partial=False,
        base_url=f"http://127.0.0.1:{port}/v1",
        api_key=secret,
        price_override=None,
        timeout_s=5.0,
        ledger_path=tmp_path / "ledger.jsonl",
        ledger_cap_usd=None,
        cache_dir=tmp_path / "cache",
        command="agent-claimcheck bench --live --out " + str(out_dir),
        date="2026-09-28",
        hardware="test-harness",
        package_version="0.0.0-test",
        dataset_sha256="a" * 64,
    )

    for name in (
        "judge-records.jsonl",
        "run.json",
        "calibration.json",
        "offline-predictions.jsonl",
        "offline-timings.jsonl",
        "lodo-models.json",
    ):
        assert (out_dir / name).exists()

    records = [
        json.loads(line)
        for line in (out_dir / "judge-records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 18  # 3 runs (2 judges + ablation) * 6 traces (3 train + 3 test)
    assert {r["detector"] for r in records} == {
        "judge:test/judge-a",
        "judge:test/judge-b",
        "judge:test/judge-a:claim-by-claim",
    }
    assert all(r["cached"] is False for r in records)  # live never reads the cache
    assert summary.calls == 18
    assert summary.cached == 0
    assert summary.total_actual_usd == pytest.approx(18 * 0.00015)
    assert summary.budget_exhausted is False
    assert "18 calls" in format_spend_summary(summary)

    run_meta = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
    judge_ids = {j["id"] for j in run_meta["judges"]}
    assert judge_ids == {MODEL_A, MODEL_B}
    assert set(run_meta["judge_wall_clock_s"]) == {
        "judge:test/judge-a",
        "judge:test/judge-b",
        "judge:test/judge-a:claim-by-claim",
    }

    calibration = json.loads((out_dir / "calibration.json").read_text(encoding="utf-8"))
    assert "judge:test/judge-a" in calibration["calibrators"]
    assert "judge:test/judge-b" in calibration["calibrators"]
    assert "judge:test/judge-a:claim-by-claim" in calibration["calibrators"]

    for name in ("judge-records.jsonl", "run.json", "calibration.json"):
        assert secret not in (out_dir / name).read_text(encoding="utf-8")
    ledger_text = (tmp_path / "ledger.jsonl").read_text(encoding="utf-8")
    assert secret not in ledger_text
    assert len(ledger_text.strip().splitlines()) == 18
    for header in _FakeORHandler.api_keys_seen:
        assert header == f"Bearer {secret}"  # the server did receive it; files never do


def test_budget_exhaustion_mid_run_still_writes_every_file(
    fake_server: int, tmp_path: Path
) -> None:
    port = fake_server
    train = load_traces("bench:train")
    test = load_traces("bench:test")
    out_dir = tmp_path / "run"

    price = Price(price_in_per_m=0.0, price_out_per_m=1_000_000.0)
    # price_in_per_m=0 makes the reservation independent of request size, so
    # every one of the 18 calls reserves exactly this much. Each call's real
    # cost settles to the fake server's tiny fixed usage.cost (0.00015), so
    # a budget of one reservation plus a sliver of headroom lets exactly the
    # first call through and exhausts the rest.
    one_call = reservation_usd("", 400, price)
    max_usd = one_call + 0.0001

    summary = run_live(
        train,
        test,
        out_dir,
        judges=[MODEL_A, MODEL_B],
        ablation_judge=MODEL_A,
        ablation_prompt="claim-by-claim",
        run_name="t2",
        max_usd=max_usd,
        concurrency=1,
        limit=3,
        allow_partial=True,
        base_url=f"http://127.0.0.1:{port}/v1",
        api_key="k",
        price_override=price,
        timeout_s=5.0,
        ledger_path=tmp_path / "ledger.jsonl",
        ledger_cap_usd=None,
        cache_dir=tmp_path / "cache",
        command="test",
        date="2026-09-28",
        hardware="test-harness",
        package_version="0.0.0-test",
        dataset_sha256="a" * 64,
    )

    assert summary.budget_exhausted is True
    records = [
        json.loads(line)
        for line in (out_dir / "judge-records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 18
    exhausted = [r for r in records if r["abstain_reason"] == "budget_exhausted"]
    assert len(exhausted) == 17  # every call after the first one exhausts the budget
    assert all(r["cost_usd"] == 0.0 for r in exhausted)
    succeeded = [r for r in records if r["abstain_reason"] != "budget_exhausted"]
    assert len(succeeded) == 1
    assert succeeded[0]["cost_usd"] == pytest.approx(0.00015)
    for name in ("run.json", "calibration.json", "offline-predictions.jsonl"):
        assert (out_dir / name).exists()


def test_preflight_refuses_without_allow_partial(fake_server: int, tmp_path: Path) -> None:
    port = fake_server
    train = load_traces("bench:train")
    test = load_traces("bench:test")
    out_dir = tmp_path / "run"

    with pytest.raises(BudgetPreflightError):
        run_live(
            train,
            test,
            out_dir,
            judges=[MODEL_A, MODEL_B],
            ablation_judge=MODEL_A,
            ablation_prompt="claim-by-claim",
            run_name="t3",
            max_usd=0.0000001,
            concurrency=1,
            limit=3,
            allow_partial=False,
            base_url=f"http://127.0.0.1:{port}/v1",
            api_key="k",
            price_override=None,
            timeout_s=5.0,
            ledger_path=tmp_path / "ledger.jsonl",
            ledger_cap_usd=None,
            cache_dir=tmp_path / "cache",
            command="test",
            date="2026-09-28",
            hardware="test-harness",
            package_version="0.0.0-test",
            dataset_sha256="a" * 64,
        )

    assert not out_dir.exists()
    assert _FakeORHandler.post_paths == []


def test_live_run_replays_byte_for_byte_via_from_recorded(fake_server: int, tmp_path: Path) -> None:
    port = fake_server
    train = load_traces("bench:train")
    test = load_traces("bench:test")
    out_dir = tmp_path / "run"
    transport = _proxy_transport(port)

    run_live(
        train,
        test,
        out_dir,
        judges=[MODEL_A],
        ablation_judge=MODEL_A,
        ablation_prompt="claim-by-claim",
        run_name="t4",
        max_usd=5.0,
        concurrency=2,
        limit=3,
        allow_partial=False,
        base_url="https://openrouter.ai/api/v1",
        api_key="k",
        price_override=None,
        timeout_s=5.0,
        ledger_path=tmp_path / "ledger.jsonl",
        ledger_cap_usd=None,
        cache_dir=tmp_path / "cache",
        command="test",
        date="2026-09-28",
        hardware="test-harness",
        package_version="0.0.0-test",
        dataset_sha256="a" * 64,
        transport=transport,
    )

    assert verify_judge_requests(out_dir) == []

    bench_json_1, bench_md_1 = build_report(out_dir)
    bench_json_2, bench_md_2 = build_report(out_dir)
    assert bench_md_1 == bench_md_2
    assert bench_json_1 == bench_json_2

    replay_out = tmp_path / "replay"
    result = runner.invoke(
        app, ["bench", "--from-recorded", str(out_dir), "--out", str(replay_out)]
    )
    assert result.exit_code == 0, result.output
    assert (replay_out / "bench.md").read_text(encoding="utf-8") == bench_md_1


def test_cli_live_requires_exactly_one_mode() -> None:
    result = runner.invoke(app, ["bench", "--offline", "--live"])
    assert result.exit_code == 2


def test_cli_dry_run_requires_live() -> None:
    result = runner.invoke(app, ["bench", "--dry-run"])
    assert result.exit_code == 2


def test_cli_live_requires_judges_and_ablation_judge() -> None:
    result = runner.invoke(app, ["bench", "--live", "--run-name", "t", "--max-usd", "1"])
    assert result.exit_code == 2
    assert "--judges" in result.output


def test_cli_price_in_and_price_out_must_be_given_together(fake_server: int) -> None:
    port = fake_server
    result = runner.invoke(
        app,
        [
            "bench",
            "--live",
            "--dry-run",
            "--judges",
            MODEL_A,
            "--ablation-judge",
            MODEL_A,
            "--run-name",
            "t",
            "--max-usd",
            "1",
            "--price-in",
            "1.0",
        ],
        env={"CLAIMCHECK_BASE_URL": f"http://127.0.0.1:{port}/v1", "CLAIMCHECK_API_KEY": "k"},
    )
    assert result.exit_code == 2
    assert "--price-in" in result.output


def test_cli_dry_run_end_to_end(fake_server: int) -> None:
    port = fake_server
    result = runner.invoke(
        app,
        [
            "bench",
            "--live",
            "--dry-run",
            "--judges",
            f"{MODEL_A},{MODEL_B}",
            "--ablation-judge",
            MODEL_A,
            "--run-name",
            "t",
            "--max-usd",
            "1",
            "--limit",
            "2",
        ],
        env={"CLAIMCHECK_BASE_URL": f"http://127.0.0.1:{port}/v1", "CLAIMCHECK_API_KEY": "k"},
    )
    assert result.exit_code == 0, result.output
    assert "grand total" in result.output
    assert _FakeORHandler.post_paths == []
