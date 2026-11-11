# SPDX-License-Identifier: Apache-2.0
"""Shared pytest fixtures and thread limits (set before NumPy import)."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_ROOT = REPO_ROOT / "fixtures"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def fixtures_root() -> Path:
    return FIXTURES_ROOT


@pytest.fixture(scope="session")
def artifacts_dir() -> Path:
    env = os.environ.get("ARTIFACTS_DIR")
    if env:
        return Path(env)
    return REPO_ROOT / "artifacts"


@pytest.fixture(scope="session")
def train_worlds(fixtures_root: Path):
    from nav_imitation_learning.env.maps import load_fixtures

    return [w for w in load_fixtures(fixtures_root) if w.family == "train"]


@pytest.fixture(scope="session")
def id_test_worlds(fixtures_root: Path):
    from nav_imitation_learning.env.maps import load_fixtures

    return [w for w in load_fixtures(fixtures_root) if w.family == "id_test"]


@pytest.fixture(scope="session")
def ood_test_worlds(fixtures_root: Path):
    from nav_imitation_learning.env.maps import load_fixtures

    return [w for w in load_fixtures(fixtures_root) if w.family == "ood_test"]


@pytest.fixture(scope="session")
def report_json(artifacts_dir: Path):
    import json

    path = artifacts_dir / "report.json"
    if not path.is_file():
        pytest.skip("report.json not present; run nil eval or set ARTIFACTS_DIR")
    return json.loads(path.read_text())
