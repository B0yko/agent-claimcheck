"""The judge HTTP client.

Error taxonomy codes only: never an exception string, a response body or a
header, anywhere in the returned `JudgeResponse`. `sleep` and `transport`
are injectable so tests exercise real retry/backoff timing without a real
clock or network.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import httpx

ErrorCode = Literal["http_429", "http_5xx", "http_4xx", "timeout", "network", "invalid_response"]

#: Total attempts = 1 initial + this many retries.
MAX_RETRIES = 2
#: `Retry-After` is honoured but clamped to this many seconds.
MAX_RETRY_AFTER_S = 10.0


@dataclass(frozen=True)
class JudgeResponse:
    """The outcome of one `complete` call. `latency_ms` includes any waits."""

    ok: bool
    content: str | None
    usage: dict[str, Any]
    error_code: ErrorCode | None
    latency_ms: float
    attempts: int


def _backoff(attempt: int) -> float:
    return 0.5 * float(2**attempt)


def _retry_after_delay(response: httpx.Response, attempt: int) -> float:
    header = response.headers.get("Retry-After")
    delay: float | None = None
    if header is not None:
        try:
            delay = float(header)
        except ValueError:
            delay = None
    if delay is None or not math.isfinite(delay) or delay < 0:
        delay = _backoff(attempt)
    return float(min(delay, MAX_RETRY_AFTER_S))


def _fail(code: ErrorCode, started: float, attempts: int) -> JudgeResponse:
    return JudgeResponse(
        ok=False,
        content=None,
        usage={},
        error_code=code,
        latency_ms=(time.perf_counter() - started) * 1000,
        attempts=attempts,
    )


class JudgeClient:
    """A synchronous HTTP/1.1 client for one judge model's chat-completions endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        timeout_s: float = 60.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout_s = timeout_s
        self._sleep = sleep
        self._client = httpx.Client(transport=transport, http2=False)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> JudgeClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def complete(self, body: dict[str, Any]) -> JudgeResponse:
        """POST `body` to `<base_url>/chat/completions`, with bounded retries."""
        url = f"{self._base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        started = time.perf_counter()
        attempts = 0
        for attempt in range(MAX_RETRIES + 1):
            attempts += 1
            try:
                response = self._client.post(
                    url, json=body, headers=headers, timeout=self._timeout_s
                )
            except httpx.TimeoutException:
                if attempt < MAX_RETRIES:
                    self._sleep(_backoff(attempt))
                    continue
                return _fail("timeout", started, attempts)
            except httpx.RequestError:
                if attempt < MAX_RETRIES:
                    self._sleep(_backoff(attempt))
                    continue
                return _fail("network", started, attempts)

            if response.status_code == 429:
                if attempt < MAX_RETRIES:
                    self._sleep(_retry_after_delay(response, attempt))
                    continue
                return _fail("http_429", started, attempts)
            if 500 <= response.status_code < 600:
                if attempt < MAX_RETRIES:
                    self._sleep(_retry_after_delay(response, attempt))
                    continue
                return _fail("http_5xx", started, attempts)
            if 400 <= response.status_code < 500:
                return _fail("http_4xx", started, attempts)

            try:
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                if not isinstance(content, str):
                    raise TypeError("message content is not a string")
                usage = data.get("usage")
                if not isinstance(usage, dict):
                    usage = {}
            except Exception:
                return _fail("invalid_response", started, attempts)

            return JudgeResponse(
                ok=True,
                content=content,
                usage=usage,
                error_code=None,
                latency_ms=(time.perf_counter() - started) * 1000,
                attempts=attempts,
            )

        return _fail("network", started, attempts)  # pragma: no cover - unreachable
