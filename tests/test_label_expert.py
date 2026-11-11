# SPDX-License-Identifier: Apache-2.0
"""Expert label query and mutant training smoke tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nav_imitation_learning.env.maps import load_fixtures
from nav_imitation_learning.eval.closed_loop import eval_suite
from nav_imitation_learning.eval.mutants import shuffle_mutant_train, zero_policy_fn
from nav_imitation_learning.eval.report import HONESTY, build_report, write_report
from nav_imitation_learning.expert.label import label
from nav_imitation_learning.policy.mlp import mlp_init
from nav_imitation_learning.train.demos import collect_all_train_demos


def test_label_from_offset_pose(id_test_worlds) -> None:
    world = id_test_worlds[0]
    pose = (world.start[0] + 0.1, world.start[1], world.start[2])
    out = label(world, pose, (0.0, 0.0), world.goal)
    assert out[0] in ("ok", "unlabelable")
    if out[0] == "ok":
        assert np.all(np.isfinite([out[1], out[2]]))


def test_shuffle_mutant_and_zero_policy(id_test_worlds) -> None:
    train = [w for w in load_fixtures(Path(__file__).resolve().parents[1] / "fixtures") if w.family == "train"]
    obs, act = collect_all_train_demos(train)
    shuf = shuffle_mutant_train(mlp_init(), obs, act)
    agg, _ = eval_suite(id_test_worlds, shuf, "mutant_shuffle")
    assert agg["n"] == 8
    zagg, zeps = eval_suite(id_test_worlds, mlp_init(), "mutant_zero", zero_policy_fn())
    assert zagg["successes"] == 0
    assert all(e["status"] in ("timeout", "collision", "success") for e in zeps)


def test_build_report_roundtrip(report_json, tmp_path) -> None:
    path = tmp_path / "r.json"
    write_report(path, report_json)
    data = report_json
    mini = {
        **data,
        "sample": False,
        "git_sha": "0" * 40,
        "honesty": HONESTY,
    }
    write_report(tmp_path / "gate.json", build_report(
        mini["git_sha"],
        "test",
        data["suites"],
        data["episodes"],
        data.get("trajectories", []),
        data["open_loop_mse"],
        data.get("dagger_selected_round", 0),
    ))
    assert (tmp_path / "r.json").is_file()
