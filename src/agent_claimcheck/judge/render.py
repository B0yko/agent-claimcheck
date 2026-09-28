"""Prompt loading and deterministic request rendering.

Nothing here makes a network call. The OpenRouter-specific extras
(`provider.require_parameters`, `reasoning.enabled`) are decided once, from
whatever a caller already knows about the model (see `openrouter_extra_body`),
and then carried as a plain dict on `JudgeSpec.extra_body` so a later replay
never has to look the model up again.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Any
from urllib.parse import urlsplit

import yaml

from agent_claimcheck.redact import DetectorView, ViewFinalClaim
from agent_claimcheck.schema import Step

#: Output field values (and step content) longer than this are cut, with a
#: `…[truncated N chars]` marker naming how many characters were dropped.
TRUNCATE_LIMIT = 2000

_BUILTIN_DIR = Path(__file__).resolve().parent / "prompts"
_BUILTIN_NAMES = ("claim-audit", "claim-by-claim")
DEFAULT_PROMPT_NAME = "claim-audit"

_FRONT_MATTER_RE = re.compile(r"\A---\n(.*?\n)---\n?(.*)\Z", re.S)


class PromptError(Exception):
    """A prompt file is malformed, or its body references an unknown placeholder."""


@dataclass(frozen=True)
class Prompt:
    """A loaded prompt: front matter plus body, and a hash of the raw file."""

    name: str
    version: int
    text: str
    sha256: str


@dataclass(frozen=True)
class JudgeSpec:
    """The shape of one judge request: what to send, not how to send it."""

    model: str
    temperature: float = 0.0
    max_tokens: int = 400
    json_mode: bool = True
    extra_body: dict[str, Any] = field(default_factory=dict)


def _read_prompt_text(path_or_builtin: str | Path) -> str:
    if isinstance(path_or_builtin, str) and path_or_builtin in _BUILTIN_NAMES:
        source = _BUILTIN_DIR / f"{path_or_builtin}.md"
    else:
        source = Path(path_or_builtin)
    try:
        return source.read_text(encoding="utf-8")
    except OSError as exc:
        raise PromptError(f"could not read prompt {path_or_builtin!r}: {exc}") from exc


def load_prompt(path_or_builtin: str | Path) -> Prompt:
    """Load a prompt by built-in name (`"claim-audit"`, `"claim-by-claim"`)
    or filesystem path. The front matter must set `name` and `version`.
    """
    text = _read_prompt_text(path_or_builtin)
    match = _FRONT_MATTER_RE.match(text)
    if not match:
        raise PromptError(f"prompt has no --- front matter: {path_or_builtin!r}")
    header_text, body = match.groups()
    try:
        header = yaml.safe_load(header_text) or {}
    except yaml.YAMLError as exc:
        raise PromptError(f"invalid front matter in {path_or_builtin!r}: {exc}") from exc
    name = header.get("name")
    version = header.get("version")
    if not isinstance(name, str) or not name:
        raise PromptError(f"prompt front matter is missing 'name': {path_or_builtin!r}")
    if not isinstance(version, int) or isinstance(version, bool):
        raise PromptError(
            f"prompt front matter is missing an integer 'version': {path_or_builtin!r}"
        )
    sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return Prompt(name=name, version=version, text=body, sha256=sha256)


def _truncate(text: str, limit: int = TRUNCATE_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"…[truncated {len(text) - limit} chars]"


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def render_steps(steps: Sequence[Step]) -> str:
    """One block per step: a `[i] kind role name` header, then whichever of
    `args:`, `ok:`, `output:`, `error:`/`content:` the step carries. Each
    top-level `output` field is serialized and truncated independently, so
    one long field never crowds out the rest of a step.
    """
    lines: list[str] = []
    for s in steps:
        header = f"[{s.i}] {s.kind} {s.role}" + (f" {s.name}" if s.name else "")
        lines.append(header)
        if s.args is not None:
            lines.append(f"args: {_truncate(_json(s.args))}")
        if s.ok is not None:
            lines.append(f"ok: {_json(s.ok)}")
        if s.output is not None:
            if isinstance(s.output, dict):
                lines.append("output:")
                for key in sorted(s.output):
                    lines.append(f"  {key}: {_truncate(_json(s.output[key]))}")
            else:
                lines.append(f"output: {_truncate(_json(s.output))}")
        if s.error is not None:
            lines.append(f"error: {_truncate(_json(s.error))}")
        if s.content is not None:
            lines.append(f"content: {_truncate(s.content)}")
    return "\n".join(lines)


def render_claims(final_claim: ViewFinalClaim) -> str:
    """The final message text plus the resolved claims, as JSON."""
    claims = [c.model_dump(mode="json") for c in final_claim.claims]
    text = final_claim.text if final_claim.text is not None else ""
    return f"final message:\n{text}\n\nresolved claims:\n{_json(claims)}"


def render_body(prompt: Prompt, *, instruction: str, steps: str, claims: str) -> str:
    """Substitute `$instruction`/`$steps`/`$claims` into a prompt body.

    A `$name` the body references that is not one of these three is a
    malformed prompt, reported as a `PromptError` naming it. `$$` renders a
    literal `$` (plain `string.Template` behaviour).
    """
    try:
        return Template(prompt.text).substitute(instruction=instruction, steps=steps, claims=claims)
    except KeyError as exc:
        raise PromptError(f"prompt references unknown placeholder ${exc.args[0]}") from exc
    except ValueError as exc:
        raise PromptError(f"prompt has invalid placeholder syntax: {exc}") from exc


def render_request(view: DetectorView, prompt: Prompt, spec: JudgeSpec) -> dict[str, Any]:
    """Build the full chat-completions request body. Deterministic."""
    body_text = render_body(
        prompt,
        instruction=view.task.instruction,
        steps=render_steps(view.steps),
        claims=render_claims(view.final_claim),
    )
    request: dict[str, Any] = {
        "model": spec.model,
        "messages": [{"role": "user", "content": body_text}],
        "temperature": spec.temperature,
        "max_tokens": spec.max_tokens,
    }
    if spec.json_mode:
        request["response_format"] = {"type": "json_object"}
    request.update(spec.extra_body)
    return request


def request_sha256(body: dict[str, Any]) -> str:
    """sha256 of the request body's canonical JSON encoding."""
    canonical = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def openrouter_extra_body(
    base_url: str, *, json_mode: bool, supports_reasoning: bool
) -> dict[str, Any]:
    """The OpenRouter-only request extras, decided once per model.

    Empty unless `base_url`'s host is exactly `openrouter.ai`. When it is:
    `provider.require_parameters` is set whenever JSON mode is requested,
    and `reasoning.enabled` is turned off for models whose `/models` listing
    reports `reasoning` in `supported_parameters`.
    """
    if urlsplit(base_url).hostname != "openrouter.ai":
        return {}
    extra: dict[str, Any] = {}
    if json_mode:
        extra["provider"] = {"require_parameters": True}
    if supports_reasoning:
        extra["reasoning"] = {"enabled": False}
    return extra
