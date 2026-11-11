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
    dagger_gate_audit,
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


def _ep(suite: str, status: str, collision: bool) -> dict:
    return {"suite": suite, "status": status, "collision": collision, "time_s": 1.0}


def _audit_eps(bc_ood: list[tuple[str, bool]]) -> list[dict]:
    eps = [_ep("bc_id", "success", False)] * 3 + [_ep("bc_id", "collision", True)] * 5
    eps += [_ep("dagger_id", "success", False)] * 4 + [_ep("dagger_id", "collision", True)] * 4
    eps += [_ep("bc_ood", s, c) for s, c in bc_ood]
    eps += [_ep("dagger_ood", "collision", True)] * 8
    return eps


def test_gate_audit_at_bc_floor_marks_ood_clauses_vacuous() -> None:
    audit = dagger_gate_audit(_audit_eps([("collision", True)] * 8))
    assert audit["ood_successes_clause_can_fail"] is False
    assert audit["ood_collisions_clause_can_fail"] is False
    assert audit["binding_clauses"] == ["dagger_id_successes >= bc_id_successes"]
    assert audit["id_clause"] == "4/8 >= 3/8"
    assert audit["original_margin_met"] is False


def test_gate_audit_off_floor_marks_ood_clauses_binding() -> None:
    audit = dagger_gate_audit(_audit_eps([("success", False)] + [("collision", True)] * 7))
    assert audit["ood_successes_clause_can_fail"] is True
    assert audit["ood_collisions_clause_can_fail"] is True
    assert len(audit["binding_clauses"]) == 3


def test_report_gate_audit_matches_records(report_json: dict) -> None:
    assert report_json["dagger_gate_audit"] == dagger_gate_audit(report_json["episodes"])


def test_tampered_gate_audit_rejected(report_json: dict, tmp_path: Path) -> None:
    bad = copy.deepcopy(report_json)
    bad["dagger_gate_audit"]["ood_successes_clause_can_fail"] = not bad["dagger_gate_audit"][
        "ood_successes_clause_can_fail"
    ]
    path = tmp_path / "bad_audit.json"
    write_report(path, bad)
    assert check_report(path, require_ci=False, expect_sha=None) == 1
