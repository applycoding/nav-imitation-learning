# SPDX-License-Identifier: Apache-2.0
"""Schema v1 JSON report writer and gate validation."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from nav_imitation_learning.constants import (
    DT,
    GOAL_TOL_XY,
    OBS_DIM,
    PLAN_INFLATION_M,
    R_MAX,
    ROBOT_RADIUS_M,
)

HONESTY = (
    "Simulation only. Synthetic maps and a scripted expert. "
    "This file is recorded output; it is not a live training run."
)

SUITE_NAMES = [
    "expert_id",
    "expert_ood",
    "bc_id",
    "bc_ood",
    "dagger_id",
    "dagger_ood",
    "mutant_shuffle",
    "mutant_zero",
]

COUNT_KEYS = (
    "n",
    "successes",
    "collisions",
    "invalids",
    "timeouts",
    "success_rate",
    "collision_rate",
)


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, allow_nan=False)
    path.write_text(text + "\n")


def build_report(
    git_sha: str,
    run_id: str,
    suites: dict[str, dict[str, Any]],
    episodes: list[dict[str, Any]],
    trajectories: list[dict[str, Any]],
    open_loop_mse: dict[str, float],
    dagger_selected_round: int,
) -> dict[str, Any]:
    for name in SUITE_NAMES:
        if name not in suites:
            raise ValueError(f"missing suite {name}")
        suites[name] = _finalize_suite(name, suites[name], episodes)
    return {
        "schema_version": 1,
        "sample": False,
        "git_sha": git_sha,
        "run_id": run_id,
        "honesty": HONESTY,
        "dagger_selected_round": dagger_selected_round,
        "seeds": {
            "train": list(range(11001, 11021)),
            "id_test": list(range(21001, 21009)),
            "ood_test": list(range(31001, 31009)),
        },
        "constants": {
            "dt": DT,
            "goal_tol_xy": GOAL_TOL_XY,
            "R_max": R_MAX,
            "obs_dim": OBS_DIM,
            "robot_radius_m": ROBOT_RADIUS_M,
            "plan_inflation_m": PLAN_INFLATION_M,
        },
        "open_loop_mse": open_loop_mse,
        "dagger_gate_audit": dagger_gate_audit(episodes),
        "suites": suites,
        "episodes": episodes,
        "trajectories": trajectories,
    }


def dagger_gate_audit(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Say which amended DAgger clauses can actually fail on this run (PLAN.md Amendment 1).

    Collisions are counted once per episode, so with n OOD episodes BC OOD
    successes are at least 0 and collisions at most n. If BC OOD has 0 successes
    and n collisions, the two OOD no-regression clauses hold for any DAgger
    result and only the in-distribution clause binds. The retired 0.25 margin
    is recomputed and recorded; it is not a ship gate.
    """
    bc_id = _recompute_agg([e for e in episodes if e.get("suite") == "bc_id"])
    bc_ood = _recompute_agg([e for e in episodes if e.get("suite") == "bc_ood"])
    dag_id = _recompute_agg([e for e in episodes if e.get("suite") == "dagger_id"])
    dag_ood = _recompute_agg([e for e in episodes if e.get("suite") == "dagger_ood"])
    n_ood = bc_ood["n"]
    succ_clause_can_fail = bool(bc_ood["successes"] > 0)
    coll_clause_can_fail = bool(bc_ood["collisions"] < n_ood)
    binding = ["dagger_id_successes >= bc_id_successes"]
    if succ_clause_can_fail:
        binding.append("dagger_ood_successes >= bc_ood_successes")
    if coll_clause_can_fail:
        binding.append("dagger_ood_collisions <= bc_ood_collisions")
    rate_bc_ood = bc_ood["successes"] / n_ood if n_ood else 0.0
    rate_dag_ood = dag_ood["successes"] / dag_ood["n"] if dag_ood["n"] else 0.0
    return {
        "ood_successes_clause_can_fail": succ_clause_can_fail,
        "ood_collisions_clause_can_fail": coll_clause_can_fail,
        "binding_clauses": binding,
        "id_clause": f"{dag_id['successes']}/{dag_id['n']} >= {bc_id['successes']}/{bc_id['n']}",
        "original_margin_met": bool(rate_dag_ood >= rate_bc_ood + 0.25),
        "original_margin_rule": "dagger_ood_rate >= bc_ood_rate + 0.25 (retired, not a ship gate)",
    }


def _finalize_suite(name: str, suite: dict[str, Any], episodes: list[dict[str, Any]]) -> dict[str, Any]:
    eps = [e for e in episodes if e.get("suite") == name]
    recomputed = _recompute_agg(eps)
    expected = suite.get("expected", _expected_for_suite(name))
    gate_ok = _gate_passes(name, recomputed, episodes)
    if name.startswith("mutant"):
        observed = "fail" if gate_ok else "pass"
    else:
        observed = "pass" if gate_ok else "fail"
    out = {**recomputed, "expected": expected, "observed": observed, "pass": observed == expected}
    return out


def _recompute_agg(eps: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(eps)
    successes = sum(1 for e in eps if e["status"] == "success")
    collisions = sum(1 for e in eps if e.get("collision"))
    invalids = sum(1 for e in eps if e["status"] == "invalid")
    timeouts = sum(1 for e in eps if e["status"] == "timeout")
    succ_times = [e["time_s"] for e in eps if e["status"] == "success"]
    return {
        "n": n,
        "successes": successes,
        "collisions": collisions,
        "invalids": invalids,
        "timeouts": timeouts,
        "success_rate": successes / n if n else 0.0,
        "collision_rate": collisions / n if n else 0.0,
        "mean_time_success_s": float(np.mean(succ_times)) if succ_times else None,
    }


def _expected_for_suite(name: str) -> str:
    if name.startswith("mutant"):
        return "fail"
    return "pass"


def _gate_passes(name: str, agg: dict[str, Any], all_eps: list[dict[str, Any]]) -> bool:
    if name == "expert_id" or name == "expert_ood":
        return bool(
            agg["successes"] == agg["n"] and agg["collisions"] == 0 and agg["invalids"] == 0
        )
    if name == "bc_id":
        return bool(agg["successes"] >= 2)
    if name == "bc_ood":
        bc_id = _recompute_agg([e for e in all_eps if e.get("suite") == "bc_id"])
        rate_id = bc_id["successes"] / bc_id["n"] if bc_id["n"] else 0.0
        rate_ood = agg["successes"] / agg["n"] if agg["n"] else 0.0
        return rate_ood <= rate_id - 0.25
    if name == "dagger_id":
        bc_id = _recompute_agg([e for e in all_eps if e.get("suite") == "bc_id"])
        return bool(agg["successes"] >= bc_id["successes"])
    if name == "dagger_ood":
        bc_id = _recompute_agg([e for e in all_eps if e.get("suite") == "bc_id"])
        bc_ood = _recompute_agg([e for e in all_eps if e.get("suite") == "bc_ood"])
        dagger_id = _recompute_agg([e for e in all_eps if e.get("suite") == "dagger_id"])
        # Amended gate (PLAN.md Amendment 1): no regression against BC out of
        # distribution, no collision regression, and no in-distribution regression.
        # This does not claim an OOD advantage over BC.
        return bool(
            agg["successes"] >= bc_ood["successes"]
            and agg["collisions"] <= bc_ood["collisions"]
            and dagger_id["successes"] >= bc_id["successes"]
        )
    if name == "mutant_shuffle":
        bc_id = _recompute_agg([e for e in all_eps if e.get("suite") == "bc_id"])
        if agg["invalids"] > 0:
            return False
        return bool(agg["successes"] <= 1 and agg["successes"] < bc_id["successes"])
    if name == "mutant_zero":
        if agg["successes"] != 0 or agg["invalids"] > 0:
            return False
        eps = [e for e in all_eps if e.get("suite") == "mutant_zero"]
        max_disp = max(_start_disp(e) for e in eps) if eps else 1.0
        return bool(max_disp <= 1e-9)
    return True


def _start_disp(ep: dict[str, Any]) -> float:
    s = ep["start"]
    f = ep["final_pose"]
    return math.hypot(f[0] - s[0], f[1] - s[1])


def check_report(path: Path, *, require_ci: bool, expect_sha: str | None) -> int:
    data = json.loads(path.read_text())
    if require_ci and data.get("sample") is True:
        print("sample flag rejected for CI", file=sys.stderr)
        return 1
    if require_ci:
        sha = data.get("git_sha", "")
        if not sha or sha == "unknown" or len(sha) != 40:
            print("invalid git_sha", file=sys.stderr)
            return 1
        if expect_sha and sha != expect_sha:
            print(f"git_sha mismatch {sha} vs {expect_sha}", file=sys.stderr)
            return 1
    eps = data.get("episodes", [])
    suites = data.get("suites", {})
    for name in SUITE_NAMES:
        if name not in suites:
            print(f"missing suite {name}", file=sys.stderr)
            return 1
        finalized = _finalize_suite(name, suites[name], eps)
        for key in COUNT_KEYS:
            if suites[name].get(key) != finalized[key]:
                print(f"stored {key} disagrees with episode records for {name}", file=sys.stderr)
                return 1
        if suites[name].get("pass") != finalized["pass"]:
            print(f"pass mismatch for {name}", file=sys.stderr)
            return 1
        if not finalized["pass"]:
            print(f"gate failed: {name}", file=sys.stderr)
            return 1
    if data.get("dagger_gate_audit") != dagger_gate_audit(eps):
        print("dagger_gate_audit disagrees with episode records", file=sys.stderr)
        return 1
    return 0


def subsample_trajectory(xy: list[list[float]]) -> list[list[float]]:
    out: list[list[float]] = []
    for i, pt in enumerate(xy):
        if i % 4 == 0:
            out.append(pt)
    if xy and (len(xy) - 1) % 4 != 0:
        if not out or out[-1] != xy[-1]:
            out.append(xy[-1])
    return out[:201]
