"""Configuration: `claimcheck.toml`, environment variables and CLI overrides.

Precedence for every setting is CLI flag > environment variable > TOML file
> built-in default. Only `.env.example` is shipped; the process environment
is read directly, and `.env` files are never parsed by this project.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_NON_SUCCESS_TYPES = ("failed", "blocked", "gave_up", "needs_input", "partial")

_ENV_BASE_URL = "CLAIMCHECK_BASE_URL"
_ENV_API_KEY = "CLAIMCHECK_API_KEY"
_ENV_OPENROUTER_KEY = "OPENROUTER_API_KEY"
_ENV_MODEL = "CLAIMCHECK_MODEL"
_ENV_MAX_USD = "CLAIMCHECK_MAX_USD"
_ENV_LEDGER = "CLAIMCHECK_LEDGER"
_ENV_LEDGER_CAP_USD = "CLAIMCHECK_LEDGER_CAP_USD"


@dataclass(frozen=True)
class GateConfig:
    verified: float = 0.80
    false_success: float = 0.20


@dataclass(frozen=True)
class JudgeConfig:
    base_url: str = DEFAULT_BASE_URL
    api_key_env: str = _ENV_API_KEY
    model: str | None = None
    prompt: str | None = None
    temperature: float = 0.0
    max_tokens: int = 400
    json_mode: bool = True
    timeout_s: float = 60.0
    concurrency: int = 8
    price_in_per_m: float | None = None
    price_out_per_m: float | None = None


@dataclass(frozen=True)
class BudgetConfig:
    max_usd: float = 1.0


@dataclass(frozen=True)
class RulesConfig:
    packs: tuple[str, ...] = ()
    non_success_types: tuple[str, ...] = DEFAULT_NON_SUCCESS_TYPES


@dataclass(frozen=True)
class ClassifierConfig:
    model: str | None = None
    calibration: str | None = None


@dataclass(frozen=True)
class Config:
    gate: GateConfig = field(default_factory=GateConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    rules: RulesConfig = field(default_factory=RulesConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    ledger: str | None = None
    ledger_cap_usd: float | None = None


def _resolve(*values: Any) -> Any:
    """First non-`None` value, in CLI > env > TOML > default order."""
    for value in values:
        if value is not None:
            return value
    return None


def _load_toml(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("rb") as fh:
        data: dict[str, Any] = tomllib.load(fh)
        return data


def _default_config_path(explicit: str | Path | None) -> Path | None:
    if explicit is not None:
        return Path(explicit)
    candidate = Path("claimcheck.toml")
    return candidate if candidate.exists() else None


def cache_dir() -> Path:
    """`$XDG_CACHE_HOME/agent-claimcheck`, or `~/.cache/agent-claimcheck`."""
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "agent-claimcheck"


def ledger_path(explicit: str | None = None) -> Path:
    """Resolve the ledger file path: `CLAIMCHECK_LEDGER` (expanduser), or
    `ledger.jsonl` in the cache directory.
    """
    value = explicit if explicit is not None else os.environ.get(_ENV_LEDGER)
    if value:
        return Path(value).expanduser()
    return cache_dir() / "ledger.jsonl"


def resolve_api_key(base_url: str, explicit: str | None = None) -> str | None:
    """`CLAIMCHECK_API_KEY`, else `OPENROUTER_API_KEY` when `base_url` is
    exactly `openrouter.ai`. The fallback key is never used for any other
    host.
    """
    key = explicit if explicit is not None else os.environ.get(_ENV_API_KEY)
    if key:
        return key
    if urlsplit(base_url).hostname == "openrouter.ai":
        return os.environ.get(_ENV_OPENROUTER_KEY)
    return None


def load_config(
    config_path: str | Path | None = None,
    *,
    cli: Mapping[str, Any] | None = None,
) -> Config:
    """Build a `Config` from CLI overrides, environment, TOML and defaults.

    `cli` is a flat mapping of dotted keys, e.g. `{"budget.max_usd": 2.0}`.
    """
    cli = cli or {}
    toml_data = _load_toml(_default_config_path(config_path))

    gate_toml = toml_data.get("gate", {})
    judge_toml = toml_data.get("judge", {})
    budget_toml = toml_data.get("budget", {})
    rules_toml = toml_data.get("rules", {})
    classifier_toml = toml_data.get("classifier", {})

    env_max_usd = os.environ.get(_ENV_MAX_USD)
    max_usd_from_env = float(env_max_usd) if env_max_usd is not None else None

    env_ledger_cap = os.environ.get(_ENV_LEDGER_CAP_USD)
    ledger_cap_from_env = float(env_ledger_cap) if env_ledger_cap is not None else None

    gate_cfg = GateConfig(
        verified=_resolve(
            cli.get("gate.verified"), gate_toml.get("verified"), GateConfig().verified
        ),
        false_success=_resolve(
            cli.get("gate.false_success"),
            gate_toml.get("false_success"),
            GateConfig().false_success,
        ),
    )

    judge_cfg = JudgeConfig(
        base_url=_resolve(
            cli.get("judge.base_url"),
            os.environ.get(_ENV_BASE_URL),
            judge_toml.get("base_url"),
            DEFAULT_BASE_URL,
        ),
        api_key_env=_resolve(
            cli.get("judge.api_key_env"), judge_toml.get("api_key_env"), _ENV_API_KEY
        ),
        model=_resolve(cli.get("judge.model"), os.environ.get(_ENV_MODEL), judge_toml.get("model")),
        prompt=_resolve(cli.get("judge.prompt"), judge_toml.get("prompt")),
        temperature=_resolve(cli.get("judge.temperature"), judge_toml.get("temperature"), 0.0),
        max_tokens=_resolve(cli.get("judge.max_tokens"), judge_toml.get("max_tokens"), 400),
        json_mode=_resolve(cli.get("judge.json_mode"), judge_toml.get("json_mode"), True),
        timeout_s=_resolve(cli.get("judge.timeout_s"), judge_toml.get("timeout_s"), 60.0),
        concurrency=_resolve(cli.get("judge.concurrency"), judge_toml.get("concurrency"), 8),
        price_in_per_m=_resolve(cli.get("judge.price_in_per_m"), judge_toml.get("price_in_per_m")),
        price_out_per_m=_resolve(
            cli.get("judge.price_out_per_m"), judge_toml.get("price_out_per_m")
        ),
    )

    budget_cfg = BudgetConfig(
        max_usd=_resolve(
            cli.get("budget.max_usd"), max_usd_from_env, budget_toml.get("max_usd"), 1.0
        ),
    )

    rules_cfg = RulesConfig(
        packs=tuple(_resolve(cli.get("rules.packs"), rules_toml.get("packs"), ())),
        non_success_types=tuple(
            _resolve(
                cli.get("rules.non_success_types"),
                rules_toml.get("non_success_types"),
                DEFAULT_NON_SUCCESS_TYPES,
            )
        ),
    )

    classifier_cfg = ClassifierConfig(
        model=_resolve(cli.get("classifier.model"), classifier_toml.get("model")),
        calibration=_resolve(cli.get("classifier.calibration"), classifier_toml.get("calibration")),
    )

    return Config(
        gate=gate_cfg,
        judge=judge_cfg,
        budget=budget_cfg,
        rules=rules_cfg,
        classifier=classifier_cfg,
        ledger=_resolve(cli.get("ledger"), os.environ.get(_ENV_LEDGER)),
        ledger_cap_usd=_resolve(cli.get("ledger_cap_usd"), ledger_cap_from_env),
    )
