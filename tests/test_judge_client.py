"""The judge HTTP client: retries, error taxonomy, latency."""

from __future__ import annotations

import json
import time
from typing import Any

import httpx
import pytest

from agent_claimcheck.judge.client import MAX_RETRY_AFTER_S, JudgeClient


def _ok_response(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": "hello"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.001},
        },
    )


def test_successful_call_returns_content_and_usage() -> None:
    transport = httpx.MockTransport(_ok_response)
    client = JudgeClient(base_url="https://example.test/v1", api_key="k", transport=transport)
    response = client.complete({"model": "m"})
    assert response.ok is True
    assert response.content == "hello"
    assert response.usage["cost"] == 0.001
    assert response.attempts == 1
    assert response.latency_ms >= 0


def test_authorization_header_sent_when_api_key_given() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("authorization")
        return _ok_response(request)

    transport = httpx.MockTransport(handler)
    client = JudgeClient(
        base_url="https://example.test/v1", api_key="secret-key", transport=transport
    )
    client.complete({"model": "m"})
    assert captured["auth"] == "Bearer secret-key"


def test_no_api_key_no_authorization_header() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("authorization")
        return _ok_response(request)

    transport = httpx.MockTransport(handler)
    client = JudgeClient(base_url="https://example.test/v1", api_key=None, transport=transport)
    client.complete({"model": "m"})
    assert captured["auth"] is None


def test_429_retried_bounded_and_reports_http_429() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429, headers={"Retry-After": "0"}, json={"error": "rate limited"})

    sleeps: list[float] = []
    transport = httpx.MockTransport(handler)
    client = JudgeClient(
        base_url="https://example.test/v1", api_key="k", transport=transport, sleep=sleeps.append
    )
    response = client.complete({"model": "m"})
    assert response.ok is False
    assert response.error_code == "http_429"
    assert response.attempts == 3
    assert calls["n"] == 3
    assert len(sleeps) == 2


def test_retry_after_clamped_to_max() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "999"})

    sleeps: list[float] = []
    transport = httpx.MockTransport(handler)
    client = JudgeClient(
        base_url="https://example.test/v1", api_key="k", transport=transport, sleep=sleeps.append
    )
    client.complete({"model": "m"})
    assert sleeps
    assert all(delay <= MAX_RETRY_AFTER_S for delay in sleeps)


def test_retry_after_non_numeric_falls_back_to_backoff() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "not-a-number"})

    sleeps: list[float] = []
    transport = httpx.MockTransport(handler)
    client = JudgeClient(
        base_url="https://example.test/v1", api_key="k", transport=transport, sleep=sleeps.append
    )
    client.complete({"model": "m"})
    assert sleeps[0] == pytest.approx(0.5)
    assert sleeps[1] == pytest.approx(1.0)


def test_5xx_retried_and_reports_http_5xx() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503)

    transport = httpx.MockTransport(handler)
    client = JudgeClient(
        base_url="https://example.test/v1", api_key="k", transport=transport, sleep=lambda s: None
    )
    response = client.complete({"model": "m"})
    assert response.error_code == "http_5xx"
    assert calls["n"] == 3


def test_4xx_not_retried() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(422)

    transport = httpx.MockTransport(handler)
    client = JudgeClient(base_url="https://example.test/v1", api_key="k", transport=transport)
    response = client.complete({"model": "m"})
    assert response.error_code == "http_4xx"
    assert calls["n"] == 1


def test_5xx_then_success_recovers() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 2:
            return httpx.Response(500)
        return _ok_response(request)

    transport = httpx.MockTransport(handler)
    client = JudgeClient(
        base_url="https://example.test/v1", api_key="k", transport=transport, sleep=lambda s: None
    )
    response = client.complete({"model": "m"})
    assert response.ok is True
    assert response.attempts == 2


def test_timeout_reports_timeout_code() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    transport = httpx.MockTransport(handler)
    client = JudgeClient(
        base_url="https://example.test/v1", api_key="k", transport=transport, sleep=lambda s: None
    )
    response = client.complete({"model": "m"})
    assert response.error_code == "timeout"
    assert response.attempts == 3


def test_network_error_reports_network_code() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    transport = httpx.MockTransport(handler)
    client = JudgeClient(
        base_url="https://example.test/v1", api_key="k", transport=transport, sleep=lambda s: None
    )
    response = client.complete({"model": "m"})
    assert response.error_code == "network"


def test_invalid_json_body_reports_invalid_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json")

    transport = httpx.MockTransport(handler)
    client = JudgeClient(base_url="https://example.test/v1", api_key="k", transport=transport)
    response = client.complete({"model": "m"})
    assert response.error_code == "invalid_response"


def test_missing_content_key_reports_invalid_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    transport = httpx.MockTransport(handler)
    client = JudgeClient(base_url="https://example.test/v1", api_key="k", transport=transport)
    response = client.complete({"model": "m"})
    assert response.error_code == "invalid_response"


def test_error_results_never_carry_exception_text_body_or_headers() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            headers={"X-Secret": "sekrit"},
            content=b'{"error": {"message": "sekrit-detail"}}',
        )

    transport = httpx.MockTransport(handler)
    client = JudgeClient(
        base_url="https://example.test/v1", api_key="k", transport=transport, sleep=lambda s: None
    )
    response = client.complete({"model": "m"})
    dumped = json.dumps(
        {
            "ok": response.ok,
            "content": response.content,
            "usage": response.usage,
            "error_code": response.error_code,
            "attempts": response.attempts,
        }
    )
    assert "sekrit" not in dumped
    assert response.error_code == "http_5xx"


def test_latency_includes_retry_waits() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "0"})

    def real_sleep(seconds: float) -> None:
        time.sleep(0.02)

    transport = httpx.MockTransport(handler)
    client = JudgeClient(
        base_url="https://example.test/v1", api_key="k", transport=transport, sleep=real_sleep
    )
    response = client.complete({"model": "m"})
    assert response.latency_ms >= 35
