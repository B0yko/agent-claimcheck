"""The LLM judge detector.

Wires the pieces in `judge/` and `ledger.py` together: render the request,
check the cache, reserve budget, call the model, settle and log the spend,
then parse the answer. Every abstain path (`no_api_key`, `unknown_price`,
`budget_exhausted`, `ledger_cap`, the client's error codes, `parse_error`)
reports the configured base rate.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import httpx

from agent_claimcheck.config import DEFAULT_BASE_URL, resolve_api_key
from agent_claimcheck.detectors.base import DetectorOutput, Reason, register_detector
from agent_claimcheck.gate import UnknownPriceError
from agent_claimcheck.judge.cache import JudgeCache, cache_key
from agent_claimcheck.judge.client import JudgeClient
from agent_claimcheck.judge.parse import ParseError, parse_judgment
from agent_claimcheck.judge.render import (
    DEFAULT_PROMPT_NAME,
    JudgeSpec,
    load_prompt,
    openrouter_extra_body,
    render_request,
    request_sha256,
)
from agent_claimcheck.ledger import (
    Budget,
    BudgetExhausted,
    Ledger,
    LedgerCapExceeded,
    LedgerEntry,
    Price,
    PriceBook,
    actual_cost_usd,
    now_iso,
    reservation_usd,
)
from agent_claimcheck.redact import DetectorView

#: Reported as `p_success` whenever the judge abstains.
DEFAULT_BASE_RATE = 0.6

#: The reason `detail` for each abstention code (`abstain_reason` keeps the code).
_ABSTAIN_DETAILS: dict[str, str] = {
    "no_api_key": "the judge did not run: no API key is set",
    "unknown_price": "the judge did not run: the model's price is unknown",
    "budget_exhausted": "the judge did not run: the run's budget is used up",
    "ledger_cap": "the judge did not run: the spending ledger's cap is reached",
    "parse_error": "the judge's answer could not be parsed",
    "http_429": "the judge API rate-limited the request",
    "http_5xx": "the judge API returned a server error",
    "http_4xx": "the judge API rejected the request",
    "timeout": "the judge call timed out",
    "network": "the judge call failed with a network error",
    "invalid_response": "the judge API returned an invalid response",
}


class JudgeDetector:
    """Scores a trace by asking an LLM to audit the final claim against the trace."""

    concurrent = True

    def __init__(
        self,
        *,
        model: str,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str | None = None,
        prompt: str | Path = DEFAULT_PROMPT_NAME,
        temperature: float = 0.0,
        max_tokens: int = 400,
        json_mode: bool = True,
        timeout_s: float = 60.0,
        extra_body: Mapping[str, Any] | None = None,
        cache: JudgeCache | None = None,
        read_cache: bool = True,
        budget: Budget | None = None,
        price_book: PriceBook | None = None,
        ledger: Ledger | None = None,
        run_id: str = "",
        base_rate: float = DEFAULT_BASE_RATE,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        if budget is not None and price_book is None:
            raise ValueError("a budget requires a price_book to size reservations")
        if ledger is not None and price_book is None:
            raise ValueError("a ledger requires a price_book to price settlements")

        self._prompt = load_prompt(prompt)
        self._model = model
        self._cache = cache
        self._read_cache = read_cache
        self._budget = budget
        self._price_book = price_book
        self._ledger = ledger
        self._run_id = run_id
        self._base_rate = base_rate

        if extra_body is None:
            supports_reasoning = (
                price_book.reasoning_supported(model) if price_book is not None else False
            )
            resolved_extra = openrouter_extra_body(
                base_url, json_mode=json_mode, supports_reasoning=supports_reasoning
            )
        else:
            resolved_extra = dict(extra_body)

        self._spec = JudgeSpec(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            extra_body=resolved_extra,
        )

        self._api_key = resolve_api_key(base_url, api_key)
        self._client = JudgeClient(
            base_url=base_url,
            api_key=self._api_key,
            timeout_s=timeout_s,
            transport=transport,
            sleep=sleep or time.sleep,
        )

        self.name = (
            f"judge:{model}"
            if self._prompt.name == DEFAULT_PROMPT_NAME
            else f"judge:{model}:{self._prompt.name}"
        )

    def close(self) -> None:
        self._client.close()

    def score(self, trace: DetectorView) -> DetectorOutput:
        body = render_request(trace, self._prompt, self._spec)
        request_json = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        key = cache_key(self._model, self._prompt.version, request_json)
        valid_steps = {s.i for s in trace.steps}

        if self._cache is not None and self._read_cache:
            hit = self._cache.get(key)
            if hit is not None:
                if self._ledger is not None:
                    self._append_ledger(
                        trace,
                        usage=hit.usage,
                        reserved=0.0,
                        actual=0.0,
                        cached=True,
                        status="cached",
                    )
                return self._finish(
                    hit.content,
                    hit.usage,
                    actual=0.0,
                    latency_ms=0.0,
                    cached=True,
                    attempts=0,
                    valid_steps=valid_steps,
                    body=body,
                )

        if self._api_key is None:
            return self._abstain("no_api_key")

        list_price = Price(price_in_per_m=0.0, price_out_per_m=0.0)
        if self._price_book is not None:
            try:
                list_price = self._price_book.list_price(self._model)
            except UnknownPriceError:
                return self._abstain("unknown_price")

        reservation = None
        if self._budget is not None and self._price_book is not None:
            amount = reservation_usd(
                request_json, self._spec.max_tokens, self._price_book.max_price(self._model)
            )
            try:
                reservation = self._budget.reserve(amount)
            except BudgetExhausted:
                return self._abstain("budget_exhausted")
            except LedgerCapExceeded:
                return self._abstain("ledger_cap")

        response = self._client.complete(body)

        if not response.ok:
            if reservation is not None and self._budget is not None:
                self._budget.release(reservation)
            if self._ledger is not None:
                self._append_ledger(
                    trace,
                    usage={},
                    reserved=reservation.amount if reservation else 0.0,
                    actual=0.0,
                    cached=False,
                    status=response.error_code or "unknown",
                )
            return self._abstain(response.error_code or "network", latency_ms=response.latency_ms)

        assert response.content is not None
        actual = actual_cost_usd(response.usage, list_price)
        if reservation is not None and self._budget is not None:
            self._budget.settle(reservation, actual)
        if self._cache is not None:
            self._cache.put(key, response.content, response.usage)
        if self._ledger is not None:
            self._append_ledger(
                trace,
                usage=response.usage,
                reserved=reservation.amount if reservation else 0.0,
                actual=actual,
                cached=False,
                status="ok",
            )

        return self._finish(
            response.content,
            response.usage,
            actual=actual,
            latency_ms=response.latency_ms,
            cached=False,
            attempts=response.attempts,
            valid_steps=valid_steps,
            body=body,
        )

    def _finish(
        self,
        content: str,
        usage: dict[str, Any],
        *,
        actual: float,
        latency_ms: float,
        cached: bool,
        attempts: int,
        valid_steps: set[int],
        body: dict[str, Any],
    ) -> DetectorOutput:
        record: dict[str, Any] = {
            "raw_response": content,
            "usage": usage,
            "cost_usd": actual,
            "latency_ms": latency_ms,
            "cached": cached,
            "prompt_name": self._prompt.name,
            "prompt_version": self._prompt.version,
            "prompt_sha256": self._prompt.sha256,
            "request_sha256": request_sha256(body),
            "attempts": attempts,
        }
        try:
            parsed = parse_judgment(content, valid_steps=valid_steps)
        except ParseError:
            # Keep the raw answer and usage: a paid call that could not be
            # parsed is exactly the record someone will want to inspect.
            return self._abstain(
                "parse_error",
                cost_usd=actual,
                latency_ms=latency_ms,
                cached=cached,
                details={**record, "parsed": None, "invalid_citation": False},
            )

        reason_step = parsed.evidence_steps[0] if parsed.evidence_steps else None
        reasons = [
            Reason(
                claim=None,
                outcome=parsed.failure_kind,
                step=reason_step,
                detail=parsed.rationale[:200],
            )
        ]
        details: dict[str, Any] = {
            "parsed": {
                "p_success": parsed.p_success,
                "failure_kind": parsed.failure_kind,
                "evidence_steps": parsed.evidence_steps,
                "rationale": parsed.rationale,
                "rationale_truncated": parsed.rationale_truncated,
                "claims": parsed.claims,
            },
            **record,
            "invalid_citation": parsed.invalid_citation,
        }
        return DetectorOutput(
            detector=self.name,
            p_success=parsed.p_success,
            abstain=False,
            reasons=reasons,
            cost_usd=actual,
            latency_ms=latency_ms,
            cached=cached,
            details=details,
        )

    def _abstain(
        self,
        reason: str,
        *,
        cost_usd: float = 0.0,
        latency_ms: float = 0.0,
        cached: bool = False,
        details: dict[str, Any] | None = None,
    ) -> DetectorOutput:
        return DetectorOutput(
            detector=self.name,
            p_success=self._base_rate,
            abstain=True,
            abstain_reason=reason,
            reasons=[
                Reason(
                    claim=None,
                    outcome=reason,
                    step=None,
                    detail=_ABSTAIN_DETAILS.get(reason, f"the judge abstained ({reason})")[:200],
                )
            ],
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            cached=cached,
            details=details or {},
        )

    def _append_ledger(
        self,
        trace: DetectorView,
        *,
        usage: dict[str, Any],
        reserved: float,
        actual: float,
        cached: bool,
        status: str,
    ) -> None:
        assert self._ledger is not None
        tokens_in = usage.get("prompt_tokens") or 0
        tokens_out = usage.get("completion_tokens") or 0
        self._ledger.append(
            LedgerEntry(
                ts=now_iso(),
                run_id=self._run_id,
                model=self._model,
                trace_id=trace.trace_id,
                prompt=self._prompt.name,
                tokens_in=int(tokens_in) if isinstance(tokens_in, (int, float)) else 0,
                tokens_out=int(tokens_out) if isinstance(tokens_out, (int, float)) else 0,
                reserved_usd=reserved,
                actual_usd=actual,
                cached=cached,
                status=status,
            )
        )


register_detector("judge", JudgeDetector)
