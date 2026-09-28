"""The dashboard's HTTP surface: security, routes, SSE and the judge panel.

Every server under test is a real `ThreadingHTTPServer` bound to
`127.0.0.1:0` (an ephemeral port), run in a background thread, so these
tests exercise the actual Host/Origin/Content-Type/size checks rather than
calling handler methods directly.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx
import pytest
from factory import message, trace

from agent_claimcheck.config import Config, JudgeConfig
from agent_claimcheck.detectors.judge import JudgeDetector
from agent_claimcheck.server.app import DashboardServer, create_server

_STATIC_DIR = Path(__file__).resolve().parents[1] / "src" / "agent_claimcheck" / "server" / "static"

_DANGEROUS_TOKENS = (
    "innerHTML",
    "outerHTML",
    "insertAdjacentHTML",
    "document.write",
    "eval(",
)


def _origin(server: DashboardServer) -> str:
    return f"http://127.0.0.1:{server.server_port}"


def _run(server: DashboardServer) -> threading.Thread:
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread


def _stop(server: DashboardServer, thread: threading.Thread) -> None:
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


@pytest.fixture
def dashboard() -> Iterator[tuple[DashboardServer, httpx.Client]]:
    """A server with no input traces, for checks that never touch a trace."""
    server = create_server([], port=0)
    thread = _run(server)
    try:
        with httpx.Client(base_url=_origin(server), timeout=5, trust_env=False) as client:
            yield server, client
    finally:
        _stop(server, thread)


@pytest.fixture
def dashboard_with_traces(tmp_path: Path) -> Iterator[tuple[DashboardServer, httpx.Client]]:
    traces = [
        trace(
            "booking-01",
            "booking",
            [message(0, "agent", "Not entirely sure this went through.")],
            text="Not entirely sure this went through.",
            claims=[("booked", {})],
        ),
        trace(
            "crm-01",
            "crm",
            [message(0, "agent", "Something might be off here.")],
            text="Something might be off here.",
            claims=[("updated", {"object": "contact", "record_id": "r1"})],
        ),
    ]
    from agent_claimcheck.schema import dump_trace

    fixture_path = tmp_path / "traces.jsonl"
    fixture_path.write_text("\n".join(dump_trace(t) for t in traces) + "\n", encoding="utf-8")

    server = create_server([str(fixture_path)], reviews=tmp_path / "reviews.jsonl", port=0)
    thread = _run(server)
    try:
        with httpx.Client(base_url=_origin(server), timeout=5, trust_env=False) as client:
            yield server, client
    finally:
        _stop(server, thread)


# -- static ------------------------------------------------------------------


def test_static_routes_only_serve_allowlisted_files(
    dashboard: tuple[DashboardServer, httpx.Client],
) -> None:
    _server, client = dashboard
    root = client.get("/")
    assert root.status_code == 200
    assert "agent-claimcheck" in root.text
    assert root.headers["content-type"].startswith("text/html")

    for path, content_type in (
        ("/static/app.js", "text/javascript"),
        ("/static/render.js", "text/javascript"),
        ("/static/format.js", "text/javascript"),
        ("/static/styles.css", "text/css"),
        ("/static/index.html", "text/html"),
    ):
        response = client.get(path)
        assert response.status_code == 200, path
        assert response.headers["content-type"].startswith(content_type)
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["content-security-policy"] == "default-src 'self'"

    for path in ("/static/pyproject.toml", "/static/../cli.py", "/static/does-not-exist.js"):
        assert client.get(path).status_code == 404
    assert client.get("/does-not-exist").status_code == 404


def test_static_javascript_never_uses_a_markup_injecting_or_eval_api() -> None:
    for name in ("app.js", "render.js", "format.js"):
        text = (_STATIC_DIR / name).read_text(encoding="utf-8")
        for token in _DANGEROUS_TOKENS:
            assert token not in text, f"{name} contains banned token {token!r}"


# -- host / origin / content-type / size checks -------------------------------


def test_host_header_must_match_an_allowed_host(
    dashboard: tuple[DashboardServer, httpx.Client],
) -> None:
    server, client = dashboard
    good = client.get("/api/state", headers={"Host": f"127.0.0.1:{server.server_port}"})
    assert good.status_code == 200
    bad = client.get("/api/state", headers={"Host": "evil.example:1234"})
    assert bad.status_code == 403


def test_post_requires_origin_content_type_and_a_bounded_body(
    dashboard: tuple[DashboardServer, httpx.Client],
) -> None:
    _server, client = dashboard
    origin = client.base_url

    no_origin = client.post("/api/judge/cancel", json={"run_id": "x"})
    assert no_origin.status_code == 403

    wrong_origin = client.post(
        "/api/judge/cancel", json={"run_id": "x"}, headers={"Origin": "http://evil.example"}
    )
    assert wrong_origin.status_code == 403

    wrong_type = client.post(
        "/api/judge/cancel",
        content=json.dumps({"run_id": "x"}),
        headers={"Origin": str(origin), "Content-Type": "text/plain"},
    )
    assert wrong_type.status_code == 415

    too_big = client.post(
        "/api/judge/cancel",
        content=b"{" + b'"run_id": "' + b"x" * 9000 + b'"}',
        headers={"Origin": str(origin), "Content-Type": "application/json"},
    )
    assert too_big.status_code == 413

    unknown_run = client.post(
        "/api/judge/cancel", json={"run_id": "does-not-exist"}, headers={"Origin": str(origin)}
    )
    assert unknown_run.status_code == 404


# -- GET never starts paid work -----------------------------------------------


def test_no_get_route_ever_scores_with_the_judge(
    dashboard_with_traces: tuple[DashboardServer, httpx.Client], monkeypatch: pytest.MonkeyPatch
) -> None:
    server, client = dashboard_with_traces

    def _forbidden(self: JudgeDetector, trace: Any) -> Any:
        pytest.fail("a GET request must never call the judge")

    monkeypatch.setattr(JudgeDetector, "score", _forbidden)

    assert client.get("/api/state").status_code == 200
    trace_id = server.ctx.store.all()[0].trace.trace_id
    assert client.get(f"/api/trace?id={trace_id}").status_code == 200
    assert client.get("/api/reviews").status_code == 200
    assert client.get("/api/judge/estimate").status_code == 200
    assert client.get("/api/judge/events?run=nope").status_code == 404


# -- review append / download -------------------------------------------------


def test_review_is_saved_as_agent_trace_v1_with_human_ground_truth_and_downloadable(
    dashboard_with_traces: tuple[DashboardServer, httpx.Client],
) -> None:
    server, client = dashboard_with_traces
    trace_id = server.ctx.store.all()[0].trace.trace_id
    origin = str(client.base_url)

    response = client.post(
        "/api/review",
        json={"trace_id": trace_id, "decision": "verified", "note": "checked manually"},
        headers={"Origin": origin},
    )
    assert response.status_code == 200
    assert response.json() == {"ok": True, "trace_id": trace_id}

    bad_decision = client.post(
        "/api/review",
        json={"trace_id": trace_id, "decision": "maybe", "note": ""},
        headers={"Origin": origin},
    )
    assert bad_decision.status_code == 400

    too_long_note = client.post(
        "/api/review",
        json={"trace_id": trace_id, "decision": "verified", "note": "x" * 1001},
        headers={"Origin": origin},
    )
    assert too_long_note.status_code == 400

    unknown_trace = client.post(
        "/api/review",
        json={"trace_id": "does-not-exist", "decision": "verified", "note": ""},
        headers={"Origin": origin},
    )
    assert unknown_trace.status_code == 404

    download = client.get("/api/reviews")
    assert download.status_code == 200
    assert "attachment" in download.headers["content-disposition"]
    assert download.headers["cache-control"] == "no-store"

    report = load_traces_report_from_text(download.text)
    assert len(report) == 1
    assert report[0]["ground_truth"]["checked_by"] == "human"
    assert report[0]["ground_truth"]["outcome"] == "success"
    assert report[0]["ground_truth"]["details"]["note"] == "checked manually"


def load_traces_report_from_text(text: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# -- queue and `serve bench:test` ---------------------------------------------


def test_serve_bench_test_state_has_a_non_empty_queue() -> None:
    server = create_server(["bench:test"], port=0)
    thread = _run(server)
    try:
        with httpx.Client(base_url=_origin(server), timeout=5, trust_env=False) as client:
            state = client.get("/api/state").json()
    finally:
        _stop(server, thread)

    assert state["total"] > 0
    assert len(state["queue"]) > 0
    p_values = [item["p_success"] for item in state["queue"]]
    distances = [abs((p if p is not None else 0.5) - 0.5) for p in p_values]
    assert distances == sorted(distances)


# -- judge estimate ------------------------------------------------------------


def test_judge_estimate_reports_not_configured_when_no_model_is_set(
    dashboard_with_traces: tuple[DashboardServer, httpx.Client],
) -> None:
    _server, client = dashboard_with_traces
    payload = client.get("/api/judge/estimate").json()
    assert payload["error"] == "not_configured"
    assert "queued" in payload


class _NotFoundHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args: Any) -> None:
        pass

    def do_GET(self) -> None:  # noqa: N802
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        self.send_response(404)
        self.end_headers()


def test_judge_estimate_reports_unknown_price_when_the_model_listing_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = ThreadingHTTPServer(("127.0.0.1", 0), _NotFoundHandler)
    fake_thread = threading.Thread(target=fake.serve_forever, daemon=True)
    fake_thread.start()
    monkeypatch.setenv("CLAIMCHECK_API_KEY", "test-key")

    from agent_claimcheck.schema import dump_trace

    fixture_path = tmp_path / "traces.jsonl"
    fixture_path.write_text(
        dump_trace(
            trace(
                "booking-01",
                "booking",
                [message(0, "agent", "Not sure.")],
                text="Not sure.",
                claims=[("booked", {})],
            )
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = Config(
        judge=JudgeConfig(base_url=f"http://127.0.0.1:{fake.server_port}/v1", model="fake/model")
    )
    server = create_server(
        [str(fixture_path)], reviews=tmp_path / "reviews.jsonl", config=cfg, port=0
    )
    thread = _run(server)
    try:
        with httpx.Client(base_url=_origin(server), timeout=5, trust_env=False) as client:
            payload = client.get("/api/judge/estimate").json()
    finally:
        _stop(server, thread)
        fake.shutdown()
        fake.server_close()
        fake_thread.join(timeout=5)

    assert payload["error"] == "unknown_price"


# -- judge run end-to-end, SSE, and budget exhaustion -------------------------


class _FakeJudgeHandler(BaseHTTPRequestHandler):
    """A local fake `/chat/completions`. Never reached once the budget is
    tiny enough that the very first reservation is refused.
    """

    def log_message(self, *_args: Any) -> None:
        pass

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        body = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "p_success": 0.9,
                                    "failure_kind": "none",
                                    "evidence_steps": [],
                                    "rationale": "looks fine",
                                }
                            )
                        }
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def fake_judge_server() -> Iterator[ThreadingHTTPServer]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeJudgeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_judge_run_with_a_tiny_budget_abstains_budget_exhausted_over_sse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_judge_server: ThreadingHTTPServer
) -> None:
    monkeypatch.setenv("CLAIMCHECK_API_KEY", "test-key")

    from agent_claimcheck.schema import dump_trace

    # domain "other" with a claim type no rule pack defines: both `rules`
    # and the classifier (trained on booking/coding/crm only) abstain, so
    # `cascade-offline` gates this "unverifiable" -- into the review queue,
    # every time, deterministically.
    fixture_path = tmp_path / "traces.jsonl"
    fixture_path.write_text(
        dump_trace(
            trace(
                "other-01",
                "other",
                [message(0, "agent", "Not sure this worked.")],
                text="Not sure this worked.",
                claims=[("custom_claim", {})],
            )
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = Config(
        judge=JudgeConfig(
            base_url=f"http://127.0.0.1:{fake_judge_server.server_port}/v1",
            model="fake/model",
            price_in_per_m=1_000_000.0,
            price_out_per_m=1_000_000.0,
        ),
    )
    server = create_server(
        [str(fixture_path)],
        reviews=tmp_path / "reviews.jsonl",
        config=cfg,
        max_usd=0.000001,
        port=0,
    )
    thread = _run(server)
    try:
        with httpx.Client(base_url=_origin(server), timeout=10, trust_env=False) as client:
            origin = str(client.base_url)
            started = client.post("/api/judge/run", json={}, headers={"Origin": origin})
            assert started.status_code == 200
            run_id = started.json()["run_id"]

            with client.stream("GET", f"/api/judge/events?run={run_id}") as response:
                assert response.status_code == 200
                assert response.headers["content-type"] == "text/event-stream"
                body = "".join(response.iter_text())
    finally:
        _stop(server, thread)

    assert "event: end" in body
    events = re.findall(r"event: progress\ndata: (\{.*\})", body)
    assert events, "expected at least one progress event"
    payloads = [json.loads(e) for e in events]
    assert all(p["verdict"] == "unverifiable" for p in payloads)
    assert "Traceback" not in body
    assert "Exception" not in body
