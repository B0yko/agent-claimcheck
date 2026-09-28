"""Pricing, reservations, budget and the ledger."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from agent_claimcheck.gate import UnknownPriceError
from agent_claimcheck.ledger import (
    Budget,
    BudgetExhausted,
    Ledger,
    LedgerCapExceeded,
    LedgerEntry,
    Price,
    PriceBook,
    actual_cost_usd,
    reservation_usd,
)

MODELS_PAYLOAD = {
    "data": [
        {
            "id": "vendor/model-a",
            "pricing": {"prompt": "0.0000001", "completion": "0.0000002"},
            "supported_parameters": ["response_format", "reasoning"],
        },
        {
            "id": "vendor/model-b",
            "pricing": {"prompt": "0.0000003", "completion": "0.0000004"},
            "supported_parameters": ["response_format"],
        },
    ]
}

ENDPOINTS_PAYLOAD = {
    "data": {
        "endpoints": [
            {"pricing": {"prompt": "0.0000001", "completion": "0.0000002"}},
            {"pricing": {"prompt": "0.00000015", "completion": "0.0000005"}},
        ]
    }
}


def _price_book(handler: Callable[[httpx.Request], httpx.Response]) -> PriceBook:
    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    return PriceBook(base_url="https://openrouter.ai/api/v1", client=client)


def test_list_price_from_models_listing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=MODELS_PAYLOAD)

    book = _price_book(handler)
    price = book.list_price("vendor/model-a")
    assert price.price_in_per_m == pytest.approx(0.1)
    assert price.price_out_per_m == pytest.approx(0.2)


def test_max_price_uses_highest_endpoint_price() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/endpoints"):
            return httpx.Response(200, json=ENDPOINTS_PAYLOAD)
        return httpx.Response(200, json=MODELS_PAYLOAD)

    book = _price_book(handler)
    price = book.max_price("vendor/model-a")
    assert price.price_in_per_m == pytest.approx(0.15)
    assert price.price_out_per_m == pytest.approx(0.5)


def test_reasoning_supported_reads_supported_parameters() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=MODELS_PAYLOAD)

    book = _price_book(handler)
    assert book.reasoning_supported("vendor/model-a") is True
    assert book.reasoning_supported("vendor/model-b") is False


def test_unknown_model_raises_unknown_price_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=MODELS_PAYLOAD)

    book = _price_book(handler)
    with pytest.raises(UnknownPriceError):
        book.list_price("vendor/does-not-exist")


def test_no_client_and_no_config_raises_unknown_price_error() -> None:
    book = PriceBook()
    with pytest.raises(UnknownPriceError):
        book.list_price("vendor/model-a")


def test_config_price_wins_over_network() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json=MODELS_PAYLOAD)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    book = PriceBook(
        client=client,
        config_prices={"vendor/model-a": Price(price_in_per_m=9.0, price_out_per_m=9.0)},
    )
    price = book.list_price("vendor/model-a")
    assert price == Price(price_in_per_m=9.0, price_out_per_m=9.0)
    assert calls["n"] == 0


def test_cli_override_wins_over_everything() -> None:
    book = PriceBook(override=Price(price_in_per_m=1.0, price_out_per_m=2.0))
    assert book.list_price("anything") == Price(price_in_per_m=1.0, price_out_per_m=2.0)
    assert book.max_price("anything") == Price(price_in_per_m=1.0, price_out_per_m=2.0)


def test_reservation_usd_formula() -> None:
    request_json = "x" * 250  # 100 estimated input tokens
    price = Price(price_in_per_m=1_000_000, price_out_per_m=2_000_000)
    amount = reservation_usd(request_json, max_tokens=10, price=price)
    # (100 * 1_000_000 + 10 * 2_000_000) / 1e6 * 1.10 = (100 + 20) * 1.10
    assert amount == pytest.approx(132.0)


def test_actual_cost_uses_usage_cost_when_present() -> None:
    price = Price(price_in_per_m=1.0, price_out_per_m=1.0)
    assert actual_cost_usd({"cost": 0.0042}, price) == pytest.approx(0.0042)


def test_actual_cost_falls_back_to_tokens_times_list_price() -> None:
    price = Price(price_in_per_m=1_000_000, price_out_per_m=2_000_000)
    cost = actual_cost_usd({"prompt_tokens": 100, "completion_tokens": 10}, price)
    assert cost == pytest.approx(100 + 20)


def test_budget_reserve_settle_release() -> None:
    budget = Budget(max_usd=1.0)
    reservation = budget.reserve(0.4)
    budget.settle(reservation, 0.3)
    assert budget.spent() == pytest.approx(0.3)
    reservation2 = budget.reserve(0.6)
    budget.release(reservation2)
    assert budget.spent() == pytest.approx(0.3)


def test_budget_refuses_when_it_would_exceed_max_usd() -> None:
    budget = Budget(max_usd=0.1)
    with pytest.raises(BudgetExhausted):
        budget.reserve(0.2)


def test_budget_concurrent_reservations_never_exceed_max_usd() -> None:
    budget = Budget(max_usd=1.0)
    results: list[str] = []
    lock = threading.Lock()

    def worker() -> None:
        try:
            reservation = budget.reserve(0.03)
        except BudgetExhausted:
            with lock:
                results.append("abstain")
            return
        time.sleep(0.005)
        budget.settle(reservation, 0.03)
        with lock:
            results.append("ok")

    threads = [threading.Thread(target=worker) for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert budget.spent() <= 1.0 + 1e-9
    ok_count = results.count("ok")
    # floor(1.0 / 0.03) = 33: no sequence of successful reservations can commit more.
    assert ok_count <= 33
    assert ok_count + results.count("abstain") == 50


def test_ledger_appends_jsonl_and_no_trace_content(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)
    ledger.append(
        LedgerEntry(
            ts="2026-01-01T00:00:00Z",
            run_id="r1",
            model="m",
            trace_id="t1",
            prompt="claim-audit",
            tokens_in=10,
            tokens_out=5,
            reserved_usd=0.01,
            actual_usd=0.009,
            cached=False,
            status="ok",
        )
    )
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert set(entry) == {
        "ts",
        "run_id",
        "model",
        "trace_id",
        "prompt",
        "tokens_in",
        "tokens_out",
        "reserved_usd",
        "actual_usd",
        "cached",
        "status",
    }
    assert "content" not in entry and "output" not in entry and "steps" not in entry


def test_ledger_total_actual_sums_entries(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)
    for cost in (0.01, 0.02, 0.03):
        ledger.append(
            LedgerEntry(
                ts="2026-01-01T00:00:00Z",
                run_id="r1",
                model="m",
                trace_id="t1",
                prompt="claim-audit",
                tokens_in=1,
                tokens_out=1,
                reserved_usd=cost,
                actual_usd=cost,
                cached=False,
                status="ok",
            )
        )
    assert ledger.total_actual() == pytest.approx(0.06)


def test_ledger_total_actual_reads_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger1 = Ledger(path)
    ledger1.append(
        LedgerEntry(
            ts="2026-01-01T00:00:00Z",
            run_id="r1",
            model="m",
            trace_id="t1",
            prompt="claim-audit",
            tokens_in=1,
            tokens_out=1,
            reserved_usd=0.5,
            actual_usd=0.5,
            cached=False,
            status="ok",
        )
    )
    ledger2 = Ledger(path)  # a fresh process re-opening the same ledger file
    assert ledger2.total_actual() == pytest.approx(0.5)


def test_budget_refuses_when_it_would_exceed_lifetime_cap(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)
    ledger.append(
        LedgerEntry(
            ts="2026-01-01T00:00:00Z",
            run_id="r1",
            model="m",
            trace_id="t1",
            prompt="claim-audit",
            tokens_in=1,
            tokens_out=1,
            reserved_usd=0.9,
            actual_usd=0.9,
            cached=False,
            status="ok",
        )
    )
    budget = Budget(max_usd=100.0, ledger=ledger, cap=1.0)
    with pytest.raises(LedgerCapExceeded):
        budget.reserve(0.2)


def test_budget_lifetime_cap_allows_call_that_fits(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "does-not-exist-ledger.jsonl")
    budget = Budget(max_usd=100.0, ledger=ledger, cap=1.0)
    reservation = budget.reserve(0.5)
    assert reservation.amount == pytest.approx(0.5)


def test_fetch_models_is_fetched_once_under_concurrent_calls() -> None:
    calls = {"n": 0}
    lock = threading.Lock()
    start = threading.Barrier(10)

    def handler(request: httpx.Request) -> httpx.Response:
        with lock:
            calls["n"] += 1
        time.sleep(0.05)  # widen the race window between the check and the write
        return httpx.Response(200, json=MODELS_PAYLOAD)

    book = _price_book(handler)

    def worker() -> None:
        start.wait()
        book.list_price("vendor/model-a")

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert calls["n"] == 1
