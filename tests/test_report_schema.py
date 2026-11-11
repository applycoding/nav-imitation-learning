# SPDX-License-Identifier: Apache-2.0
"""Schema v1 report validation and gate recomputation."""

from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

from nav_imitation_learning.eval.report import (
    HONESTY,
    SUITE_NAMES,
    _finalize_suite,
    check_report,
    write_report,
)

REPO = Path(__file__).resolve().parent.parent


def test_report_schema_fields(report_json: dict) -> None:
    assert report_json["schema_version"] == 1
    assert report_json["sample"] is False
    assert report_json["honesty"] == HONESTY
    assert len(report_json["suites"]) == 8
    for name in SUITE_NAMES:
        assert name in report_json["suites"]
    eps = report_json["episodes"]
    # Eight suites (six policy suites plus two mutants) times 8 seeds.
    assert len(eps) == 64
    for name in SUITE_NAMES:
        assert sum(1 for e in eps if e["suite"] == name) == 8
    for name in SUITE_NAMES:
        finalized = _finalize_suite(name, copy.deepcopy(report_json["suites"][name]), eps)
        stored = report_json["suites"][name]
        assert stored["pass"] == finalized["pass"]
        assert stored["successes"] == finalized["successes"]


def test_mutated_counts_rejected(report_json: dict, tmp_path: Path) -> None:
    bad = copy.deepcopy(report_json)
    bad["suites"]["bc_id"]["successes"] = 99
    path = tmp_path / "bad.json"
    write_report(path, bad)
    finalized = _finalize_suite("bc_id", bad["suites"]["bc_id"], bad["episodes"])
    assert finalized["successes"] != bad["suites"]["bc_id"]["successes"]
    assert check_report(path, require_ci=False, expect_sha=None) == 1


def test_mutated_rate_and_n_rejected(report_json: dict, tmp_path: Path) -> None:
    for key, value in (("success_rate", 1.0), ("n", 7), ("collisions", -1)):
        bad = copy.deepcopy(report_json)
        bad["suites"]["dagger_ood"][key] = value
        path = tmp_path / f"bad_{key}.json"
        write_report(path, bad)
        assert check_report(path, require_ci=False, expect_sha=None) == 1


def test_dagger_regression_fails_gate(report_json: dict, tmp_path: Path) -> None:
    """A DAgger policy that does worse than BC out of distribution must fail."""
    bad = copy.deepcopy(report_json)
    bc_ood = [e for e in bad["episodes"] if e["suite"] == "bc_ood"]
    dag_ood = [e for e in bad["episodes"] if e["suite"] == "dagger_ood"]
    for b, d in zip(bc_ood, dag_ood, strict=True):
        b["status"] = "success"
        b["collision"] = False
        d["status"] = "collision"
        d["collision"] = True
    finalized = _finalize_suite("dagger_ood", bad["suites"]["dagger_ood"], bad["episodes"])
    assert finalized["pass"] is False


def test_sample_flag_rejected_for_ci(report_json: dict, tmp_path: Path) -> None:
    bad = copy.deepcopy(report_json)
    bad["sample"] = True
    path = tmp_path / "sample.json"
    write_report(path, bad)
    assert check_report(path, require_ci=True, expect_sha=bad["git_sha"]) == 1


def test_check_report_valid(artifacts_dir: Path, report_json: dict) -> None:
    path = artifacts_dir / "report.json"
    code = check_report(path, require_ci=False, expect_sha=None)
    assert code == 0


def test_open_loop_mse_finite(report_json: dict) -> None:
    for key in ("bc_id", "dagger_id", "shuffle_id"):
        assert isinstance(report_json["open_loop_mse"][key], (int, float))


def test_coverage_fail_under_raises() -> None:
    """Deliberately uncovered stub must fail when coverage floor is raised."""
    stub = REPO / "tests" / "cov_uncovered_stub"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(stub / "test_nothing.py"),
            f"--cov={stub.name}",
            "--cov-fail-under=80",
            "-q",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
