# SPDX-License-Identifier: Apache-2.0
"""README and NOTICE honesty boundary checks."""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

REQUIRED_PHRASES = (
    "Simulation only",
    "Synthetic maps",
    "scripted expert",
    "does not train",
)

FORBIDDEN = (
    "certified",
    "production driving policy",
    "real robot logs",
    "foundation model",
)


def _read_text(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def _overclaim_lines(text: str) -> list[str]:
    hits: list[str] = []
    for line in text.splitlines():
        lower = line.lower()
        if "not " in lower or "does not" in lower or "is not" in lower:
            continue
        for phrase in FORBIDDEN:
            if phrase in lower:
                hits.append(line)
                break
    return hits


def test_readme_honesty() -> None:
    text = _read_text("README.md")
    for phrase in REQUIRED_PHRASES:
        assert phrase.lower() in text.lower()
    assert _overclaim_lines(text) == []


def test_notice_honesty() -> None:
    text = _read_text("NOTICE")
    assert "Arpan Bom" in text
    assert "numpy" in text.lower() or "NumPy" in text
    assert _overclaim_lines(text) == []


def test_no_typographic_dash_in_authored_python() -> None:
    pattern = re.compile("[\u2013\u2014\u2212]")
    for path in REPO.rglob("*.py"):
        if "node_modules" in path.parts or ".venv" in path.parts:
            continue
        if path.is_relative_to(REPO / "tests" / "cov_uncovered_stub"):
            continue
        text = path.read_text(encoding="utf-8")
        assert not pattern.search(text), f"typographic dash in {path}"


def test_spdx_headers_on_authored_sources() -> None:
    import subprocess

    files = subprocess.check_output(["git", "ls-files"], cwd=REPO, text=True).split()
    exts = (".py", ".ts", ".html", ".css", ".toml", ".in", ".yml")
    missing = [
        f
        for f in files
        if f.endswith(exts) and "SPDX-License-Identifier: Apache-2.0" not in (REPO / f).read_text(encoding="utf-8")
    ]
    assert missing == []
