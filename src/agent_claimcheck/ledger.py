"""Pricing, reservations and the spend ledger for live judge calls.

Three things live here on purpose: `PriceBook` (what a model costs, and
whether it can take a `reasoning` parameter, both read from OpenRouter's
`/models` listing), `Budget` (a thread-safe reserve/settle/release guard
against a per-run cap and an optional lifetime cap), and `Ledger` (the
append-only spend log). None of the three ever sees trace content.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from agent_claimcheck.config import DEFAULT_BASE_URL
from agent_claimcheck.gate import UnknownPriceError

__all__ = [
    "Budget",
    "BudgetExhausted",
    "LedgerCapExceeded",
    "Ledger",
    "LedgerEntry",
    "Price",
    "PriceBook",
    "Reservation",
    "actual_cost_usd",
    "reservation_usd",
]


@dataclass(frozen=True)
class Price:
    """A model's per-million-token price, input and output."""

    price_in_per_m: float
    price_out_per_m: float


class PriceBook:
    """Model prices and `reasoning`-parameter support.

    Resolution order per model: an explicit `override` (`--price-in`/
    `--price-out`) first, then a configured price, then OpenRouter's
    `/models` listing (cached once per process). `max_price` additionally
    consults `/models/{author}/{slug}/endpoints` and takes the highest
    per-endpoint price, for a conservative reservation.
    """

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str | None = None,
        config_prices: Mapping[str, Price] | None = None,
        override: Price | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._config_prices = dict(config_prices or {})
        self._override = override
        self._client = client
        self._models: list[dict[str, Any]] | None = None
        self._endpoints_cache: dict[str, list[dict[str, Any]]] = {}

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

    def _fetch_models(self) -> list[dict[str, Any]]:
        if self._models is not None:
            return self._models
        if self._client is None:
            self._models = []
            return self._models
        try:
            response = self._client.get(f"{self._base_url}/models", headers=self._headers())
            response.raise_for_status()
            data = response.json().get("data", [])
        except (httpx.HTTPError, ValueError):
            data = []
        self._models = data if isinstance(data, list) else []
        return self._models

    def _listing(self, model: str) -> dict[str, Any] | None:
        for entry in self._fetch_models():
            if isinstance(entry, dict) and entry.get("id") == model:
                return entry
        return None

    def _fetch_endpoints(self, model: str) -> list[dict[str, Any]]:
        if model in self._endpoints_cache:
            return self._endpoints_cache[model]
        endpoints: list[dict[str, Any]] = []
        if self._client is not None and "/" in model:
            try:
                response = self._client.get(
                    f"{self._base_url}/models/{model}/endpoints", headers=self._headers()
                )
                response.raise_for_status()
                raw = response.json().get("data", {}).get("endpoints", [])
                if isinstance(raw, list):
                    endpoints = raw
            except (httpx.HTTPError, ValueError):
                endpoints = []
        self._endpoints_cache[model] = endpoints
        return endpoints

    def reasoning_supported(self, model: str) -> bool:
        """Whether the model's listing declares a `reasoning` parameter."""
        listing = self._listing(model)
        if listing is None:
            return False
        supported = listing.get("supported_parameters") or []
        return "reasoning" in supported

    def list_price(self, model: str) -> Price:
        """The model's top-level list price. Raises `UnknownPriceError`."""
        if self._override is not None:
            return self._override
        if model in self._config_prices:
            return self._config_prices[model]
        listing = self._listing(model)
        if listing is None:
            raise UnknownPriceError(f"no price configured for model {model!r}")
        pricing = listing.get("pricing") or {}
        try:
            return Price(
                price_in_per_m=float(pricing["prompt"]) * 1_000_000,
                price_out_per_m=float(pricing["completion"]) * 1_000_000,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise UnknownPriceError(f"no price configured for model {model!r}") from exc

    def max_price(self, model: str) -> Price:
        """The highest per-endpoint price seen for the model, for reservations."""
        if self._override is not None:
            return self._override
        if model in self._config_prices:
            return self._config_prices[model]
        base = self.list_price(model)
        max_in, max_out = base.price_in_per_m, base.price_out_per_m
        for endpoint in self._fetch_endpoints(model):
            pricing = endpoint.get("pricing") or {}
            try:
                max_in = max(max_in, float(pricing["prompt"]) * 1_000_000)
                max_out = max(max_out, float(pricing["completion"]) * 1_000_000)
            except (KeyError, TypeError, ValueError):
                continue
        return Price(price_in_per_m=max_in, price_out_per_m=max_out)


def reservation_usd(request_json: str, max_tokens: int, price: Price) -> float:
    """Conservative pre-call cost estimate, at `price` (normally the max
    per-endpoint price): request bytes / 2.5 as a token-count proxy for the
    input, `max_tokens` for the output, plus a 10% safety margin.
    """
    estimated_tokens_in = len(request_json) / 2.5
    raw = estimated_tokens_in * price.price_in_per_m + max_tokens * price.price_out_per_m
    return raw / 1_000_000 * 1.10


def actual_cost_usd(usage: Mapping[str, Any], list_price: Price) -> float:
    """`usage.cost` when present, else usage tokens priced at `list_price`."""
    cost = usage.get("cost")
    if isinstance(cost, (int, float)) and not isinstance(cost, bool):
        return float(cost)
    tokens_in = usage.get("prompt_tokens") or 0
    tokens_out = usage.get("completion_tokens") or 0
    if not isinstance(tokens_in, (int, float)) or isinstance(tokens_in, bool):
        tokens_in = 0
    if not isinstance(tokens_out, (int, float)) or isinstance(tokens_out, bool):
        tokens_out = 0
    cost = tokens_in * list_price.price_in_per_m + tokens_out * list_price.price_out_per_m
    return cost / 1_000_000


class BudgetExhausted(Exception):
    """A reservation would push this run's spend past `max_usd`."""


class LedgerCapExceeded(Exception):
    """A reservation would push the lifetime ledger total past the cap."""


@dataclass
class Reservation:
    """An outstanding hold against a `Budget`. Settle or release it once."""

    amount: float
    _closed: bool = field(default=False, repr=False)


class Budget:
    """Thread-safe reserve/settle/release against `max_usd`, and optionally
    against a lifetime `cap` tracked in `ledger`.
    """

    def __init__(
        self, max_usd: float, ledger: Ledger | None = None, cap: float | None = None
    ) -> None:
        self.max_usd = max_usd
        self._ledger = ledger
        self._cap = cap
        self._lock = threading.Lock()
        self._spent = 0.0
        self._outstanding = 0.0

    def reserve(self, amount: float) -> Reservation:
        with self._lock:
            if self._spent + self._outstanding + amount > self.max_usd:
                raise BudgetExhausted(
                    f"reserving ${amount:.6f} would exceed the ${self.max_usd:.2f} run budget"
                )
            if self._cap is not None:
                already = self._ledger.total_actual() if self._ledger is not None else 0.0
                if already + self._outstanding + amount > self._cap:
                    raise LedgerCapExceeded(
                        f"reserving ${amount:.6f} would exceed the ${self._cap:.2f} lifetime cap"
                    )
            self._outstanding += amount
            return Reservation(amount=amount)

    def settle(self, reservation: Reservation, actual: float) -> None:
        with self._lock:
            if reservation._closed:
                raise ValueError("reservation already settled or released")
            reservation._closed = True
            self._outstanding -= reservation.amount
            self._spent += actual

    def release(self, reservation: Reservation) -> None:
        with self._lock:
            if reservation._closed:
                return
            reservation._closed = True
            self._outstanding -= reservation.amount

    def spent(self) -> float:
        with self._lock:
            return self._spent


@dataclass(frozen=True)
class LedgerEntry:
    """One line of spend. Never carries trace content."""

    ts: str
    run_id: str
    model: str
    trace_id: str
    prompt: str
    tokens_in: int
    tokens_out: int
    reserved_usd: float
    actual_usd: float
    cached: bool
    status: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "ts": self.ts,
            "run_id": self.run_id,
            "model": self.model,
            "trace_id": self.trace_id,
            "prompt": self.prompt,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "reserved_usd": self.reserved_usd,
            "actual_usd": self.actual_usd,
            "cached": self.cached,
            "status": self.status,
        }


class Ledger:
    """Append-only JSONL spend log, flushed and fsynced on every write."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._total_actual = self._read_initial_total()

    def _read_initial_total(self) -> float:
        total = 0.0
        try:
            with self._path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        total += float(json.loads(line).get("actual_usd") or 0.0)
                    except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
                        continue
        except OSError:
            pass
        return total

    def append(self, entry: LedgerEntry) -> None:
        line = json.dumps(entry.as_dict(), sort_keys=True, ensure_ascii=False)
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            self._total_actual += entry.actual_usd

    def total_actual(self) -> float:
        with self._lock:
            return self._total_actual


def now_iso() -> str:
    """An RFC 3339 UTC timestamp for ledger entries."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
