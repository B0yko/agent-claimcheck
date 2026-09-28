"""The judge detector: cache, budget, ledger, api-key fallback, parsing,
and an end-to-end pass against a real local fake server.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx
import pytest
from factory import tool_call, tool_result, trace

from agent_claimcheck.detectors.judge import JudgeDetector
from agent_claimcheck.judge.cache import JudgeCache, cache_key
from agent_claimcheck.judge.render import JudgeSpec, Prompt, load_prompt, render_request
from agent_claimcheck.ledger import Budget, Ledger, Price, PriceBook
from agent_claimcheck.redact import DetectorView, detector_view


def _view(text: str = "Booked the room.") -> DetectorView:
    t = trace(
        "t1",
        "booking",
        [
            tool_call(0, "calendar.create_event", {"title": "sync"}),
            tool_result(
                1,
                "calendar.create_event",
                ok=True,
                output={"event_id": "e1", "status": "confirmed"},
            ),
        ],
        text=text,
        claims=[("booked", {})],
    )
    return detector_view(t)


def _expected_body(
    view: DetectorView, model: str, prompt_name: str = "claim-audit"
) -> tuple[dict[str, Any], Prompt]:
    prompt = load_prompt(prompt_name)
    spec = JudgeSpec(model=model, temperature=0.0, max_tokens=400, json_mode=True, extra_body={})
    return render_request(view, prompt, spec), prompt


def _valid_content(p_success: float = 0.9) -> str:
    return json.dumps(
        {
            "p_success": p_success,
            "failure_kind": "none",
            "evidence_steps": [1],
            "rationale": "the receipt confirms the booking",
        }
    )


def _ok_handler(content: str, usage: dict[str, Any]) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content}}], "usage": usage},
        )

    return handler


def test_detector_name_uses_default_prompt() -> None:
    detector = JudgeDetector(model="vendor/m", api_key="k", extra_body={})
    assert detector.name == "judge:vendor/m"


def test_detector_name_includes_non_default_prompt() -> None:
    detector = JudgeDetector(model="vendor/m", api_key="k", prompt="claim-by-claim", extra_body={})
    assert detector.name == "judge:vendor/m:claim-by-claim"


def test_successful_score_returns_parsed_output_and_details() -> None:
    handler = _ok_handler(
        _valid_content(0.87), {"prompt_tokens": 40, "completion_tokens": 12, "cost": 0.00021}
    )
    transport = httpx.MockTransport(handler)
    detector = JudgeDetector(model="m", api_key="k", extra_body={}, transport=transport)
    output = detector.score(_view())

    assert output.abstain is False
    assert output.p_success == pytest.approx(0.87)
    assert output.cost_usd == pytest.approx(0.00021)
    assert output.cached is False
    assert output.detector == "judge:m"
    assert output.reasons[0].outcome == "none"
    details = output.details
    assert details["prompt_name"] == "claim-audit"
    assert details["prompt_version"] == 1
    assert len(details["prompt_sha256"]) == 64
    assert len(details["request_sha256"]) == 64
    assert details["invalid_citation"] is False
    assert details["attempts"] == 1
    assert details["raw_response"] == _valid_content(0.87)


def test_parse_error_abstains_but_still_records_cost() -> None:
    handler = _ok_handler(
        "not valid json", {"prompt_tokens": 3, "completion_tokens": 2, "cost": 0.0005}
    )
    transport = httpx.MockTransport(handler)
    detector = JudgeDetector(model="m", api_key="k", extra_body={}, transport=transport)
    output = detector.score(_view())

    assert output.abstain is True
    assert output.abstain_reason == "parse_error"
    assert output.reasons[0].detail == "the judge's answer could not be parsed"
    assert output.cost_usd == pytest.approx(0.0005)
    assert output.p_success == 0.6  # base rate
    assert output.details["raw_response"] == "not valid json"
    assert output.details["usage"]["cost"] == 0.0005
    assert output.details["parsed"] is None
    assert len(output.details["request_sha256"]) == 64


def test_client_error_code_abstains_with_that_reason() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422)

    transport = httpx.MockTransport(handler)
    detector = JudgeDetector(model="m", api_key="k", extra_body={}, transport=transport)
    output = detector.score(_view())

    assert output.abstain is True
    assert output.abstain_reason == "http_4xx"
    assert output.reasons[0].detail == "the judge API rejected the request"
    assert output.cost_usd == 0.0


def test_no_api_key_abstains_before_any_http_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("CLAIMCHECK_API_KEY", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no HTTP call should happen without an api key")

    transport = httpx.MockTransport(handler)
    detector = JudgeDetector(
        model="m",
        base_url="https://example.test/v1",
        api_key=None,
        extra_body={},
        transport=transport,
    )
    output = detector.score(_view())

    assert output.abstain is True
    assert output.abstain_reason == "no_api_key"
    assert output.reasons[0].detail == "the judge did not run: no API key is set"
    assert output.p_success == 0.6


def test_fallback_openrouter_key_sent_to_openrouter_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "fallback-secret")
    monkeypatch.delenv("CLAIMCHECK_API_KEY", raising=False)
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": _valid_content()}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    transport = httpx.MockTransport(handler)
    detector = JudgeDetector(
        model="m",
        base_url="https://openrouter.ai/api/v1",
        api_key=None,
        extra_body={},
        transport=transport,
    )
    output = detector.score(_view())
    assert output.abstain is False
    assert captured["auth"] == "Bearer fallback-secret"


def test_fallback_openrouter_key_never_sent_to_another_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "fallback-secret")
    monkeypatch.delenv("CLAIMCHECK_API_KEY", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no HTTP call should happen: the fallback key must not resolve here")

    transport = httpx.MockTransport(handler)
    detector = JudgeDetector(
        model="m",
        base_url="https://example.test/v1",
        api_key=None,
        extra_body={},
        transport=transport,
    )
    output = detector.score(_view())
    assert output.abstain_reason == "no_api_key"


def test_cache_hit_returns_cached_true_cost_zero_no_http_call(tmp_path: Path) -> None:
    view = _view()
    body, prompt = _expected_body(view, "m")
    request_json = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    key = cache_key("m", prompt.version, request_json)

    cache = JudgeCache(tmp_path)
    cache.put(key, _valid_content(0.95), {"prompt_tokens": 1, "completion_tokens": 1})

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("a cache hit must never make an HTTP call")

    transport = httpx.MockTransport(handler)
    detector = JudgeDetector(
        model="m", api_key="k", cache=cache, extra_body={}, transport=transport
    )
    output = detector.score(view)

    assert output.cached is True
    assert output.cost_usd == 0.0
    assert output.abstain is False
    assert output.p_success == pytest.approx(0.95)


def test_read_cache_false_ignores_hit_but_still_writes(tmp_path: Path) -> None:
    view = _view()
    body, prompt = _expected_body(view, "m")
    request_json = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    key = cache_key("m", prompt.version, request_json)

    cache = JudgeCache(tmp_path)
    cache.put(key, _valid_content(0.1), {"prompt_tokens": 1, "completion_tokens": 1})

    handler = _ok_handler(
        _valid_content(0.8), {"prompt_tokens": 5, "completion_tokens": 5, "cost": 0.001}
    )
    transport = httpx.MockTransport(handler)
    detector = JudgeDetector(
        model="m", api_key="k", cache=cache, read_cache=False, extra_body={}, transport=transport
    )
    output = detector.score(view)

    assert output.cached is False
    assert output.p_success == pytest.approx(0.8)

    hit = cache.get(key)
    assert hit is not None
    assert hit.content == _valid_content(0.8)


def test_budget_exhausted_abstains_without_http_call() -> None:
    price_book = PriceBook(override=Price(price_in_per_m=1e12, price_out_per_m=1e12))
    budget = Budget(max_usd=0.0000001)

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("budget_exhausted must refuse before any HTTP call")

    transport = httpx.MockTransport(handler)
    detector = JudgeDetector(
        model="m",
        api_key="k",
        extra_body={},
        budget=budget,
        price_book=price_book,
        transport=transport,
    )
    output = detector.score(_view())
    assert output.abstain is True
    assert output.abstain_reason == "budget_exhausted"
    assert output.p_success == 0.6


def test_ledger_cap_abstains_without_http_call(tmp_path: Path) -> None:
    from agent_claimcheck.ledger import LedgerEntry

    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)
    ledger.append(
        LedgerEntry(
            ts="2026-01-01T00:00:00Z",
            run_id="r",
            model="m",
            trace_id="t",
            prompt="claim-audit",
            tokens_in=1,
            tokens_out=1,
            reserved_usd=0.99,
            actual_usd=0.99,
            cached=False,
            status="ok",
        )
    )
    # Priced so the reservation clears the $100 run budget but crosses the
    # $1.00 lifetime cap (already at $0.99 from the pre-existing entry).
    price_book = PriceBook(override=Price(price_in_per_m=1000.0, price_out_per_m=1000.0))
    budget = Budget(max_usd=100.0, ledger=ledger, cap=1.0)

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("ledger_cap must refuse before any HTTP call")

    transport = httpx.MockTransport(handler)
    detector = JudgeDetector(
        model="m",
        api_key="k",
        extra_body={},
        budget=budget,
        price_book=price_book,
        ledger=ledger,
        transport=transport,
    )
    output = detector.score(_view())
    assert output.abstain is True
    assert output.abstain_reason == "ledger_cap"


def test_unknown_price_with_budget_abstains_before_http_call() -> None:
    price_book = PriceBook()  # no override, no config, no client: every model is unknown
    budget = Budget(max_usd=100.0)

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("unknown_price must refuse before any HTTP call")

    transport = httpx.MockTransport(handler)
    detector = JudgeDetector(
        model="vendor/unknown-model",
        api_key="k",
        extra_body={},
        budget=budget,
        price_book=price_book,
        transport=transport,
    )
    output = detector.score(_view())
    assert output.abstain is True
    assert output.abstain_reason == "unknown_price"
    assert output.p_success == 0.6


def test_unknown_price_book_and_ledger_without_budget_abstains_before_http_call(
    tmp_path: Path,
) -> None:
    price_book = PriceBook()  # no override, no config, no client: every model is unknown
    ledger = Ledger(tmp_path / "ledger.jsonl")

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("unknown_price must refuse before any HTTP call")

    transport = httpx.MockTransport(handler)
    detector = JudgeDetector(
        model="vendor/unknown-model",
        api_key="k",
        extra_body={},
        price_book=price_book,
        ledger=ledger,
        transport=transport,
    )
    output = detector.score(_view())
    assert output.abstain is True
    assert output.abstain_reason == "unknown_price"
    ledger_path = tmp_path / "ledger.jsonl"
    assert not ledger_path.exists() or ledger_path.read_text() == ""


def test_ledger_records_a_line_per_call_with_no_trace_content(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)
    price_book = PriceBook(override=Price(price_in_per_m=1.0, price_out_per_m=1.0))
    handler = _ok_handler(
        _valid_content(0.9), {"prompt_tokens": 5, "completion_tokens": 5, "cost": 0.0002}
    )
    transport = httpx.MockTransport(handler)
    detector = JudgeDetector(
        model="m",
        api_key="k",
        extra_body={},
        ledger=ledger,
        price_book=price_book,
        transport=transport,
    )
    detector.score(_view("a very secret trace-only phrase, never logged"))

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["status"] == "ok"
    assert entry["actual_usd"] == pytest.approx(0.0002)
    assert "a very secret trace-only phrase" not in lines[0]


def test_budget_requires_price_book() -> None:
    with pytest.raises(ValueError):
        JudgeDetector(model="m", api_key="k", extra_body={}, budget=Budget(max_usd=1.0))


def test_ledger_requires_price_book(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        JudgeDetector(
            model="m", api_key="k", extra_body={}, ledger=Ledger(tmp_path / "ledger.jsonl")
        )


class _FakeJudgeHandler(BaseHTTPRequestHandler):
    call_count = 0

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        type(self).call_count += 1
        if type(self).call_count == 1:
            self.send_response(429)
            self.send_header("Retry-After", "0")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        payload = json.dumps(
            {
                "choices": [{"message": {"content": _valid_content(0.93)}}],
                "usage": {"prompt_tokens": 40, "completion_tokens": 12, "cost": 0.00021},
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass


def test_end_to_end_against_a_real_local_fake_server() -> None:
    _FakeJudgeHandler.call_count = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeJudgeHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        detector = JudgeDetector(
            model="m",
            base_url=f"http://127.0.0.1:{port}/v1",
            api_key="k",
            json_mode=True,
            extra_body={},
            sleep=lambda s: None,
        )
        output = detector.score(_view())

        assert output.abstain is False
        assert output.p_success == pytest.approx(0.93)
        assert output.cost_usd == pytest.approx(0.00021)
        assert output.details["attempts"] == 2
        assert _FakeJudgeHandler.call_count == 2
    finally:
        server.shutdown()
        thread.join(timeout=5)
