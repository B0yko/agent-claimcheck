"""The review dashboard's HTTP server: `agent-claimcheck serve`.

A stdlib `ThreadingHTTPServer` (ADR 0001) serving a small JSON API and a
handful of static files (plain ES modules, no build step) over the traces
named on the command line. At startup, every input trace is scored once
with `Checker("cascade-offline")` (or loaded from `--results`); the review
queue is whatever came out `unverifiable`. The judge panel scores that queue
on demand, through the same `JudgeDetector`/`Budget`/`Ledger` machinery
`Checker` itself uses for a live run, and never on a GET request.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qs, urlsplit

import httpx

from agent_claimcheck.calibration import CalibratorSet
from agent_claimcheck.checker import Checker, CheckResult
from agent_claimcheck.config import Config, cache_dir, load_config, resolve_api_key
from agent_claimcheck.detectors.judge import JudgeDetector
from agent_claimcheck.gate import Thresholds, UnknownPriceError, confidence, gate
from agent_claimcheck.judge.render import (
    DEFAULT_PROMPT_NAME,
    JudgeSpec,
    Prompt,
    load_prompt,
    openrouter_extra_body,
    render_request,
)
from agent_claimcheck.ledger import Price, PriceBook, reservation_usd
from agent_claimcheck.schema import Trace, load_traces
from agent_claimcheck.server.batch import ACTIVE_STATUSES, AlreadyRunning, JudgeBatch
from agent_claimcheck.server.state import (
    TraceStore,
    append_review,
    build_store,
    inspector_payload,
)

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_STATIC_ALLOWLIST: dict[str, str] = {
    "index.html": "text/html; charset=utf-8",
    "app.js": "text/javascript; charset=utf-8",
    "render.js": "text/javascript; charset=utf-8",
    "format.js": "text/javascript; charset=utf-8",
    "styles.css": "text/css; charset=utf-8",
}

_MAX_BODY_BYTES = 8192
_MAX_NOTE_CHARS = 1000
_SSE_POLL_S = 0.05

#: A page open in another tab can send unlimited blind GETs to this local
#: server (Origin gating is POST-only, by design). Cap how many
#: connections can be in flight at once so that can't spawn unbounded
#: server threads; a connection past the cap is closed immediately rather
#: than queued.
_MAX_CONCURRENT_CONNECTIONS = 32

#: Bounds how long a connection may sit idle mid-request (e.g. a request
#: line that never arrives), so a worker thread -- and the connection slot
#: above -- can't be held forever.
_REQUEST_TIMEOUT_S = 30.0

_ServerRequest = socket.socket | tuple[bytes, socket.socket]


@dataclass
class JudgeContext:
    """What the dashboard needs to run and estimate live judge calls."""

    detector: JudgeDetector
    price_book: PriceBook
    prompt: Prompt
    spec: JudgeSpec
    concurrency: int


@dataclass
class DashboardContext:
    """Everything a request handler needs, attached to the server instance."""

    store: TraceStore
    calibrators: CalibratorSet | None
    thresholds: Thresholds
    reviews_path: Path
    static_dir: Path
    allowed_hosts: frozenset[str]
    allowed_origins: frozenset[str]
    batch: JudgeBatch
    judge: JudgeContext | None
    judge_available: bool
    inputs: tuple[str, ...]


def _first(query: dict[str, list[str]], key: str) -> str | None:
    values = query.get(key)
    return values[0] if values else None


def _load_results(path: str | Path) -> list[CheckResult]:
    results: list[CheckResult] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            results.append(CheckResult.model_validate(json.loads(line)))
    return results


def _build_judge_context(cfg: Config, max_usd: float) -> tuple[JudgeContext | None, bool]:
    if not cfg.judge.model:
        return None, False
    try:
        judge_checker = Checker("judge", config=cfg, max_usd=max_usd)
    except ValueError:
        return None, False
    detector = judge_checker.detector
    if not isinstance(detector, JudgeDetector):  # pragma: no cover - defensive
        return None, False

    configured_key = os.environ.get(cfg.judge.api_key_env) if cfg.judge.api_key_env else None
    api_key = resolve_api_key(cfg.judge.base_url, configured_key)
    available = api_key is not None

    config_prices = None
    if cfg.judge.price_in_per_m is not None and cfg.judge.price_out_per_m is not None:
        price = Price(cfg.judge.price_in_per_m, cfg.judge.price_out_per_m)
        config_prices = {cfg.judge.model: price}
    price_book = PriceBook(
        base_url=cfg.judge.base_url,
        api_key=api_key,
        config_prices=config_prices,
        client=None if config_prices is not None else httpx.Client(),
    )
    prompt = load_prompt(cfg.judge.prompt or DEFAULT_PROMPT_NAME)
    spec = JudgeSpec(
        model=cfg.judge.model,
        temperature=cfg.judge.temperature,
        max_tokens=cfg.judge.max_tokens,
        json_mode=cfg.judge.json_mode,
        extra_body=openrouter_extra_body(
            cfg.judge.base_url,
            json_mode=cfg.judge.json_mode,
            supports_reasoning=price_book.reasoning_supported(cfg.judge.model),
        ),
    )
    return (
        JudgeContext(
            detector=detector,
            price_book=price_book,
            prompt=prompt,
            spec=spec,
            concurrency=cfg.judge.concurrency,
        ),
        available,
    )


def _score_with_judge(
    store: TraceStore,
    judge: JudgeDetector,
    calibrators: CalibratorSet | None,
    thresholds: Thresholds,
    trace_id: str,
) -> dict[str, Any]:
    record = store.get(trace_id)
    if record is None:
        return {
            "trace_id": trace_id,
            "verdict": None,
            "p_success": None,
            "cost_usd": 0.0,
            "error": "unknown_trace",
        }

    raw = judge.score(record.view)
    p_calibrated = calibrators.apply(raw.detector, raw) if calibrators is not None else None
    final = (
        raw.model_copy(update={"p_calibrated": p_calibrated}) if p_calibrated is not None else raw
    )
    p_used = final.p_calibrated if final.p_calibrated is not None else final.p_success
    verdict = gate(p_used, final.abstain, thresholds)

    updated = record.result.model_copy(
        update={
            "detector": final.detector,
            "verdict": verdict,
            "p_success": p_used,
            "p_success_raw": final.p_success,
            "calibrated": final.p_calibrated is not None,
            "abstain": final.abstain,
            "abstain_reason": final.abstain_reason,
            "confidence": confidence(verdict, p_used),
            "reasons": list(final.reasons) or list(record.result.reasons),
            "detectors": [final],
            "cost_usd": final.cost_usd,
            "latency_ms": final.latency_ms,
            "cached": final.cached,
        }
    )
    store.apply_judge_result(trace_id, result=updated, judge_output=final)
    return {
        "trace_id": trace_id,
        "verdict": verdict,
        "p_success": p_used,
        "cost_usd": final.cost_usd,
        "error": final.abstain_reason if final.abstain else None,
    }


def _not_configured_score(trace_id: str) -> dict[str, Any]:
    return {
        "trace_id": trace_id,
        "verdict": None,
        "p_success": None,
        "cost_usd": 0.0,
        "error": "not_configured",
    }


class DashboardServer(ThreadingHTTPServer):
    """A `ThreadingHTTPServer` carrying the dashboard's shared context.

    Bounded: at most `_MAX_CONCURRENT_CONNECTIONS` requests are served at
    once, so a same-machine page with an open tab can't spawn unbounded
    server threads by firing blind GETs; a connection past the cap is
    closed immediately rather than queued behind the rest.
    """

    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        handler_cls: type[BaseHTTPRequestHandler],
        ctx: DashboardContext,
    ) -> None:
        super().__init__(address, handler_cls)
        self.ctx = ctx
        self._connection_slots = threading.BoundedSemaphore(_MAX_CONCURRENT_CONNECTIONS)

    def process_request(self, request: _ServerRequest, client_address: Any) -> None:
        if not self._connection_slots.acquire(blocking=False):
            self.close_request(request)
            return
        super().process_request(request, client_address)

    def process_request_thread(self, request: _ServerRequest, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._connection_slots.release()


def create_server(
    inputs: Sequence[str],
    *,
    results: str | Path | None = None,
    reviews: str | Path = Path("claimcheck-reviews.jsonl"),
    host: str = "127.0.0.1",
    port: int = 8765,
    max_usd: float = 1.0,
    config: str | Path | Config | None = None,
) -> DashboardServer:
    """Build (without starting) the dashboard's HTTP server."""
    cfg = config if isinstance(config, Config) else load_config(config)

    traces: list[Trace] = []
    for one_input in inputs:
        traces.extend(load_traces(one_input))

    checker = Checker("cascade-offline", config=cfg, max_usd=max_usd)
    if results is not None:
        results_by_id: Mapping[str, CheckResult] = {r.trace_id: r for r in _load_results(results)}
    else:
        results_by_id = {r.trace_id: r for r in checker.check(traces)}

    store = build_store(traces, results_by_id, checker.extractor)

    judge_ctx, judge_available = _build_judge_context(cfg, max_usd)

    def score(trace_id: str) -> dict[str, Any]:
        assert judge_ctx is not None
        return _score_with_judge(
            store, judge_ctx.detector, checker.calibrators, checker.thresholds, trace_id
        )

    run_state_path = cache_dir() / "dashboard" / "judge-run.json"
    batch = JudgeBatch(run_state_path, score if judge_ctx is not None else _not_configured_score)

    ctx = DashboardContext(
        store=store,
        calibrators=checker.calibrators,
        thresholds=checker.thresholds,
        reviews_path=Path(reviews),
        static_dir=_STATIC_DIR,
        allowed_hosts=frozenset(),
        allowed_origins=frozenset(),
        batch=batch,
        judge=judge_ctx,
        judge_available=judge_available,
        inputs=tuple(inputs),
    )
    server = DashboardServer((host, port), Handler, ctx)

    # The actual bound port (relevant when `port=0` asked for an ephemeral
    # one) is only known once the socket exists, so the Host/Origin
    # allowlist is filled in after construction, from `server.server_port`.
    bound_port = server.server_port
    allowed_hosts = frozenset(
        {f"127.0.0.1:{bound_port}", f"localhost:{bound_port}", f"{host}:{bound_port}"}
    )
    ctx.allowed_hosts = allowed_hosts
    ctx.allowed_origins = frozenset(f"http://{h}" for h in allowed_hosts)
    return server


def serve(
    inputs: Sequence[str],
    *,
    results: str | Path | None = None,
    reviews: str | Path = Path("claimcheck-reviews.jsonl"),
    host: str = "127.0.0.1",
    port: int = 8765,
    max_usd: float = 1.0,
    config: str | Path | None = None,
) -> None:
    """Build and run the dashboard's HTTP server (blocks until stopped)."""
    server = create_server(
        inputs,
        results=results,
        reviews=reviews,
        host=host,
        port=port,
        max_usd=max_usd,
        config=config,
    )
    print(f"http://{host}:{port}")
    try:
        server.serve_forever()
    finally:
        server.server_close()


class Handler(BaseHTTPRequestHandler):
    """Routes every request; the dashboard's whole security surface lives here.

    Deliberately HTTP/1.0 (the stdlib default): every response closes the
    connection, so a request rejected before its body is fully read (a
    missing/wrong Origin, the wrong Content-Type, an over-size body) never
    desynchronises a reused keep-alive connection.

    `timeout` bounds how long a connection may sit idle mid-request (e.g. a
    request line that never arrives): past it, `handle_one_request` gives
    up and the connection closes, freeing its worker thread and connection
    slot instead of holding both forever.
    """

    timeout = _REQUEST_TIMEOUT_S

    def log_message(self, format_: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        pass

    def _ctx(self) -> DashboardContext:
        return cast(DashboardServer, self.server).ctx

    # -- low-level response helpers ----------------------------------

    def _security_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'")
        self.send_header("Referrer-Policy", "no-referrer")

    def _send_plain(self, status: int, text: str) -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self._security_headers()
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict[str, Any], *, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self._security_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _not_found(self) -> None:
        self._send_plain(404, "not found")

    # -- request-wide checks -------------------------------------------

    def _check_host(self) -> bool:
        host_header = self.headers.get("Host", "")
        if host_header not in self._ctx().allowed_hosts:
            self._send_plain(403, "host not allowed")
            return False
        return True

    def _read_json_body(self) -> dict[str, Any] | None:
        origin = self.headers.get("Origin")
        if not origin or origin not in self._ctx().allowed_origins:
            self._send_json({"error": "origin_not_allowed"}, status=403)
            return None

        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip()
        if content_type != "application/json":
            self._send_json({"error": "unsupported_media_type"}, status=415)
            return None

        # A request carrying both `Content-Length` and `Transfer-Encoding`
        # (or `Transfer-Encoding` alone) is framed ambiguously (RFC 7230,
        # section 3.3.3): reject it rather than trusting `Content-Length`
        # and silently ignoring the other header.
        if self.headers.get("Transfer-Encoding") is not None:
            self._send_json({"error": "unsupported_transfer_encoding"}, status=400)
            return None

        length_header = self.headers.get("Content-Length")
        try:
            length = int(length_header) if length_header is not None else -1
        except ValueError:
            length = -1
        if length < 0:
            self._send_json({"error": "invalid_length"}, status=400)
            return None
        if length > _MAX_BODY_BYTES:
            self._send_json({"error": "payload_too_large"}, status=413)
            return None

        raw = self.rfile.read(length) if length else b""
        try:
            parsed: Any = json.loads(raw) if raw else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json({"error": "invalid_json"}, status=400)
            return None
        if not isinstance(parsed, dict):
            self._send_json({"error": "invalid_json"}, status=400)
            return None
        return cast(dict[str, Any], parsed)

    # -- routing ---------------------------------------------------------

    def do_GET(self) -> None:
        if not self._check_host():
            return
        try:
            parsed = urlsplit(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)
            if path == "/":
                self._serve_static("index.html")
            elif path.startswith("/static/"):
                self._serve_static(path[len("/static/") :])
            elif path == "/api/state":
                self._get_state()
            elif path == "/api/trace":
                self._get_trace(query)
            elif path == "/api/reviews":
                self._get_reviews()
            elif path == "/api/judge/estimate":
                self._get_judge_estimate()
            elif path == "/api/judge/events":
                self._get_judge_events(query)
            else:
                self._not_found()
        except Exception:
            self._send_json({"error": "internal_error"}, status=500)

    def do_POST(self) -> None:
        if not self._check_host():
            return
        try:
            path = urlsplit(self.path).path
            if path == "/api/review":
                self._post_review()
            elif path == "/api/judge/run":
                self._post_judge_run()
            elif path == "/api/judge/cancel":
                self._post_judge_cancel()
            else:
                self._not_found()
        except Exception:
            self._send_json({"error": "internal_error"}, status=500)

    # -- static ------------------------------------------------------------

    def _serve_static(self, name: str) -> None:
        content_type = _STATIC_ALLOWLIST.get(name)
        if content_type is None:
            self._not_found()
            return
        try:
            body = (self._ctx().static_dir / name).read_bytes()
        except OSError:
            self._not_found()
            return
        self.send_response(200)
        self._security_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # -- GET /api/* ----------------------------------------------------

    def _get_state(self) -> None:
        ctx = self._ctx()
        queue = [
            {
                "trace_id": r.trace.trace_id,
                "domain": r.trace.task.domain,
                "p_success": r.result.p_success,
                "top_reason": r.result.reasons[0].detail if r.result.reasons else "",
            }
            for r in ctx.store.queue()
        ]
        self._send_json(
            {
                "inputs": list(ctx.inputs),
                "total": len(ctx.store.all()),
                "overview": ctx.store.overview(),
                "queue": queue,
                "judge_configured": ctx.judge_available,
            }
        )

    def _get_trace(self, query: dict[str, list[str]]) -> None:
        trace_id = _first(query, "id")
        if not trace_id:
            self._send_json({"error": "missing_id"}, status=400)
            return
        record = self._ctx().store.get(trace_id)
        if record is None:
            self._send_json({"error": "unknown_trace"}, status=404)
            return
        self._send_json(inspector_payload(record))

    def _get_reviews(self) -> None:
        try:
            body = self._ctx().reviews_path.read_bytes()
        except OSError:
            body = b""
        self.send_response(200)
        self._security_headers()
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Content-Disposition", 'attachment; filename="claimcheck-reviews.jsonl"')
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _get_judge_estimate(self) -> None:
        ctx = self._ctx()
        queue = ctx.store.queue()
        payload: dict[str, Any] = {"queued": len(queue)}
        if ctx.judge is None:
            payload["error"] = "not_configured"
            self._send_json(payload)
            return
        try:
            price = ctx.judge.price_book.max_price(ctx.judge.spec.model)
        except UnknownPriceError:
            payload["error"] = "unknown_price"
            self._send_json(payload)
            return
        total = 0.0
        for record in queue:
            body = render_request(record.view, ctx.judge.prompt, ctx.judge.spec)
            request_json = json.dumps(
                body, sort_keys=True, ensure_ascii=False, separators=(",", ":")
            )
            total += reservation_usd(request_json, ctx.judge.spec.max_tokens, price)
        payload["worst_case_usd"] = round(total, 6)
        self._send_json(payload)

    def _get_judge_events(self, query: dict[str, list[str]]) -> None:
        run_id = _first(query, "run")
        batch = self._ctx().batch
        if not run_id or batch.snapshot(run_id) is None:
            self._send_json({"error": "unknown_run"}, status=404)
            return

        self.send_response(200)
        self._security_headers()
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        sent = 0
        try:
            while True:
                snapshot = batch.snapshot(run_id)
                if snapshot is None:
                    break
                results = snapshot["results"]
                while sent < len(results):
                    item = results[sent]
                    sent += 1
                    event = {
                        "done": snapshot["done"],
                        "total": snapshot["total"],
                        "spent_usd": snapshot["spent_usd"],
                        "trace_id": item.get("trace_id"),
                        "verdict": item.get("verdict"),
                    }
                    self.wfile.write(f"event: progress\ndata: {json.dumps(event)}\n\n".encode())
                    self.wfile.flush()
                if snapshot["status"] not in ACTIVE_STATUSES and sent >= len(results):
                    break
                time.sleep(_SSE_POLL_S)
            self.wfile.write(b"event: end\ndata: {}\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    # -- POST /api/* -----------------------------------------------------

    def _post_review(self) -> None:
        body = self._read_json_body()
        if body is None:
            return
        trace_id = body.get("trace_id")
        decision = body.get("decision")
        note = body.get("note", "")
        if not isinstance(trace_id, str) or not trace_id:
            self._send_json({"error": "invalid_trace_id"}, status=400)
            return
        if decision not in ("verified", "false_success"):
            self._send_json({"error": "invalid_decision"}, status=400)
            return
        if not isinstance(note, str) or len(note) > _MAX_NOTE_CHARS:
            self._send_json({"error": "invalid_note"}, status=400)
            return
        record = self._ctx().store.get(trace_id)
        if record is None:
            self._send_json({"error": "unknown_trace"}, status=404)
            return
        try:
            append_review(self._ctx().reviews_path, record, decision=decision, note=note)
        except OSError:
            self._send_json({"error": "save_failed"}, status=500)
            return
        self._send_json({"ok": True, "trace_id": trace_id})

    def _post_judge_run(self) -> None:
        body = self._read_json_body()
        if body is None:
            return
        ctx = self._ctx()
        if ctx.judge is None or not ctx.judge_available:
            self._send_json({"error": "no_key"}, status=400)
            return
        trace_ids = [r.trace.trace_id for r in ctx.store.queue()]
        try:
            snapshot = ctx.batch.start(trace_ids, concurrency=ctx.judge.concurrency)
        except AlreadyRunning as exc:
            self._send_json({"error": "already_running", "run_id": exc.run_id}, status=409)
            return
        self._send_json({"run_id": snapshot["run_id"]})

    def _post_judge_cancel(self) -> None:
        body = self._read_json_body()
        if body is None:
            return
        run_id = body.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            self._send_json({"error": "invalid_run_id"}, status=400)
            return
        snapshot = self._ctx().batch.cancel(run_id)
        if snapshot is None:
            self._send_json({"error": "unknown_run"}, status=404)
            return
        self._send_json({"run_id": snapshot["run_id"], "status": snapshot["status"]})
