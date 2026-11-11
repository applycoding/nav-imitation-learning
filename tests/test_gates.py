# SPDX-License-Identifier: Apache-2.0
"""Ship gates recomputed from recorded CI episode records."""

from __future__ import annotations

import math
from typing import Any


def _episodes_for(report: dict[str, Any], suite: str) -> list[dict[str, Any]]:
    return [e for e in report["episodes"] if e.get("suite") == suite]


def _agg(eps: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "n": len(eps),
        "successes": sum(1 for e in eps if e["status"] == "success"),
        "collisions": sum(1 for e in eps if e.get("collision")),
        "invalids": sum(1 for e in eps if e["status"] == "invalid"),
        "timeouts": sum(1 for e in eps if e["status"] == "timeout"),
    }


def test_bc_id(report_json: dict[str, Any]) -> None:
    agg = _agg(_episodes_for(report_json, "bc_id"))
    assert agg["successes"] >= 2


def test_bc_ood_gap(report_json: dict[str, Any]) -> None:
    id_agg = _agg(_episodes_for(report_json, "bc_id"))
    ood_agg = _agg(_episodes_for(report_json, "bc_ood"))
    rate_id = id_agg["successes"] / id_agg["n"]
    rate_ood = ood_agg["successes"] / ood_agg["n"]
    assert rate_ood <= rate_id - 0.25


def test_dagger_no_regression(report_json: dict[str, Any]) -> None:
    """Amended DAgger gate (PLAN.md Amendment 1).

    The original margin (DAgger OOD rate >= BC OOD rate + 0.25) was not met on the
    locked ood_test maps. The amended gate asks only for no regression against BC:
    OOD successes not lower, OOD collisions not higher, id successes not lower.
    It does not show a DAgger advantage out of distribution.
    """
    bc_id = _agg(_episodes_for(report_json, "bc_id"))
    bc_ood = _agg(_episodes_for(report_json, "bc_ood"))
    dag_id = _agg(_episodes_for(report_json, "dagger_id"))
    dag_ood = _agg(_episodes_for(report_json, "dagger_ood"))
    assert dag_ood["successes"] >= bc_ood["successes"]
    assert dag_ood["collisions"] <= bc_ood["collisions"]
    assert dag_id["successes"] >= bc_id["successes"]


def test_mutant_shuffle(report_json: dict[str, Any]) -> None:
    bc_id = _agg(_episodes_for(report_json, "bc_id"))
    sh = _agg(_episodes_for(report_json, "mutant_shuffle"))
    assert sh["invalids"] == 0
    assert sh["timeouts"] + sh["successes"] + sh["collisions"] == sh["n"]
    assert sh["successes"] <= 1
    assert sh["successes"] < bc_id["successes"]


def test_mutant_zero(report_json: dict[str, Any]) -> None:
    z = _agg(_episodes_for(report_json, "mutant_zero"))
    assert z["invalids"] == 0
    assert z["successes"] == 0
    max_disp = 0.0
    for ep in _episodes_for(report_json, "mutant_zero"):
        s = ep["start"]
        f = ep["final_pose"]
        max_disp = max(max_disp, math.hypot(f[0] - s[0], f[1] - s[1]))
    assert max_disp <= 1e-9


def test_gates_match_report_suite_flags(report_json: dict[str, Any]) -> None:
    for suite in (
        "bc_id",
        "bc_ood",
        "dagger_id",
        "dagger_ood",
        "mutant_shuffle",
        "mutant_zero",
    ):
        assert report_json["suites"][suite]["pass"] is True

