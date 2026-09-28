"""The wheel must carry a byte-identical copy of the canonical schema."""

from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_packaged_schema_is_byte_identical_to_repo_schema(tmp_path: Path) -> None:
    subprocess.run(
        ["uv", "build", "--wheel", "--offline", "--out-dir", str(tmp_path)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    wheels = list(tmp_path.glob("*.whl"))
    assert len(wheels) == 1

    canonical = (REPO_ROOT / "schemas" / "agent-trace-v1.json").read_bytes()

    with zipfile.ZipFile(wheels[0]) as wheel:
        packaged = wheel.read("agent_claimcheck/schemas/agent-trace-v1.json")

    assert packaged == canonical
