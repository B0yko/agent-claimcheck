"""The on-disk judge response cache.

One file per request, keyed by a hash of the model, the prompt version and
the canonical request JSON, under `<cache_dir>/judge/`. Writes are atomic
(tempfile in the same directory, then `os.replace`), so a crash mid-write
never leaves a corrupt cache entry visible.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def cache_key(model: str, prompt_version: int, request_json: str) -> str:
    """sha256 of `model + "\\n" + prompt_version + "\\n" + request_json`."""
    payload = f"{model}\n{prompt_version}\n{request_json}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CacheEntry:
    """What a cache hit restores: the raw response content and its usage."""

    content: str
    usage: dict[str, Any]


class JudgeCache:
    """Reads and writes cached judge responses under `<cache_dir>/judge/`."""

    def __init__(self, cache_dir: str | Path) -> None:
        self._dir = Path(cache_dir) / "judge"

    def _path(self, key: str) -> Path:
        return self._dir / f"{key}.json"

    def get(self, key: str) -> CacheEntry | None:
        try:
            raw = self._path(key).read_text(encoding="utf-8")
        except OSError:
            return None
        try:
            data = json.loads(raw)
            return CacheEntry(content=data["content"], usage=data.get("usage") or {})
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

    def put(self, key: str, content: str, usage: dict[str, Any]) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {"content": content, "usage": usage}, sort_keys=True, ensure_ascii=False
        )
        fd, tmp_name = tempfile.mkstemp(dir=self._dir, prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_name, self._path(key))
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
