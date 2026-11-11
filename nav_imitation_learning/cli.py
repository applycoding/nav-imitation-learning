# SPDX-License-Identifier: Apache-2.0
"""Console script `nil` for generate, train, eval, and report."""

from __future__ import annotations

import argparse
import copy
import os
import sys
import uuid
from pathlib import Path

import numpy as np
import numpy.typing as npt

from nav_imitation_learning.constants import (
    ID_TEST_SEEDS,
    OOD_TEST_SEEDS,
    TRAIN_SEEDS,
)
from nav_imitation_learning.env.maps import (
    Family,
    World,
    generate_map,
    load_fixtures,
    write_fixture,
)
from nav_imitation_learning.eval.closed_loop import EpisodeDict, eval_suite
from nav_imitation_learning.eval.mutants import shuffle_mutant_train, zero_policy_fn
from nav_imitation_learning.eval.report import (
    build_report,
    check_report,
    subsample_trajectory,
    write_report,
)
from nav_imitation_learning.expert.rollout import expert_episode
from nav_imitation_learning.policy.mlp import load_params, mlp_forward, mlp_init, save_params
from nav_imitation_learning.train.bc import train_bc
from nav_imitation_learning.train.dagger import train_dagger
from nav_imitation_learning.train.demos import collect_all_train_demos


def main() -> None:
    parser = argparse.ArgumentParser(prog="nil")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate")
    gen.add_argument("--out", type=Path, required=True)

    tbc = sub.add_parser("train-bc")
    tbc.add_argument("--fixtures", type=Path, required=True)
    tbc.add_argument("--init-out", type=Path, required=True)
    tbc.add_argument("--out", type=Path, required=True)

    td = sub.add_parser("train-dagger")
    td.add_argument("--fixtures", type=Path, required=True)
    td.add_argument("--init", type=Path, required=True)
    td.add_argument("--bc", type=Path, required=True)
    td.add_argument("--out", type=Path, required=True)

    ev = sub.add_parser("eval")
    ev.add_argument("--fixtures", type=Path, required=True)
    ev.add_argument("--bc", type=Path, required=True)
    ev.add_argument("--dagger", type=Path, required=True)
    ev.add_argument("--out", type=Path, required=True)
    ev.add_argument("--git-sha", default=os.environ.get("GITHUB_SHA", "0" * 40))

    rp = sub.add_parser("report")
    rp.add_argument("--check", type=Path, required=True)
    rp.add_argument("--require-ci", action="store_true")
    rp.add_argument("--expect-sha", default=None)

    args = parser.parse_args()
    if args.command == "generate":
        cmd_generate(args.out)
    elif args.command == "train-bc":
        cmd_train_bc(args.fixtures, args.init_out, args.out)
    elif args.command == "train-dagger":
        cmd_train_dagger(args.fixtures, args.init, args.bc, args.out)
    elif args.command == "eval":
        code = cmd_eval(args.fixtures, args.bc, args.dagger, args.out, args.git_sha)
        sys.exit(code)
    elif args.command == "report":
        code = check_report(args.check, require_ci=args.require_ci, expect_sha=args.expect_sha)
        sys.exit(code)


def cmd_generate(out: Path) -> None:
    seeds: list[tuple[int, Family]] = []
    for s in TRAIN_SEEDS:
        seeds.append((s, "train"))
    for s in ID_TEST_SEEDS:
        seeds.append((s, "id_test"))
    for s in OOD_TEST_SEEDS:
        seeds.append((s, "ood_test"))
    for seed, family in seeds:
        world = generate_map(seed, family)
        if world is None:
            raise RuntimeError(f"generator failed for seed {seed}")
        write_fixture(world, out / family / str(seed))
        print(f"generated {family}/{seed}")


def _split_worlds(fixtures: Path) -> tuple[list[World], list[World], list[World]]:
    worlds = load_fixtures(fixtures)
    train = [w for w in worlds if w.family == "train"]
    id_test = [w for w in worlds if w.family == "id_test"]
    ood = [w for w in worlds if w.family == "ood_test"]
    return train, id_test, ood


def cmd_train_bc(fixtures: Path, init_out: Path, out: Path) -> None:
    train, _, _ = _split_worlds(fixtures)
    init = mlp_init()
    init_out.parent.mkdir(parents=True, exist_ok=True)
    save_params(str(init_out), init)
    obs, act = collect_all_train_demos(train)
    params = train_bc(copy.deepcopy(init), obs, act)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_params(str(out), params)


def cmd_train_dagger(fixtures: Path, init_path: Path, bc_path: Path, out: Path) -> None:
    train, _, _ = _split_worlds(fixtures)
    init = load_params(str(init_path))
    obs, act = collect_all_train_demos(train)
    dagger_params, meta = train_dagger(init, train, obs, act)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_params(str(out), dagger_params, extra={"dagger_selected_round": meta["dagger_selected_round"]})


def _open_loop_mse(
    params: dict[str, npt.NDArray[np.float64]],
    id_worlds: list[World],
) -> float:
    from nav_imitation_learning.constants import OMEGA_MAX, V_MAX
    from nav_imitation_learning.env.rays import observe
    from nav_imitation_learning.expert.label import label

    errs: list[float] = []
    for world in id_worlds:
        pose = world.start
        prev = (0.0, 0.0)
        for _ in range(800):
            import math

            from nav_imitation_learning.constants import GOAL_TOL_XY

            if math.hypot(world.goal[0] - pose[0], world.goal[1] - pose[1]) <= GOAL_TOL_XY:
                break
            o = observe(world.occupancy, pose, world.goal, prev)
            lab = label(world, pose, prev, world.goal)
            if lab[0] != "ok":
                break
            pre, _ = mlp_forward(params, o.astype(np.float64).reshape(1, -1))
            tgt = np.array([[lab[1] / V_MAX, lab[2] / OMEGA_MAX]])
            errs.append(float(np.mean((pre - tgt) ** 2)))
            from nav_imitation_learning.env.collision import swept_step

            pose, st = swept_step(pose, lab[1], lab[2], 0.05, world.rectangles)
            prev = (lab[1], lab[2])
            if st != "ok":
                break
    return float(np.mean(errs)) if errs else 0.0


def cmd_eval(fixtures: Path, bc_path: Path, dagger_path: Path, out: Path, git_sha: str) -> int:
    train, id_test, ood = _split_worlds(fixtures)
    bc_params = load_params(str(bc_path))
    dagger_data = np.load(str(dagger_path))
    dagger_params = load_params(str(dagger_path))
    dagger_round = int(dagger_data["dagger_selected_round"]) if "dagger_selected_round" in dagger_data else 0

    init = mlp_init()
    obs, act = collect_all_train_demos(train)
    shuffle_params = shuffle_mutant_train(init, obs, act)

    suites: dict[str, dict[str, object]] = {}
    episodes: list[EpisodeDict] = []
    trajectories: list[dict[str, object]] = []

    from nav_imitation_learning.eval.closed_loop import _aggregate

    for name, worlds, fn in [
        ("expert_id", id_test, None),
        ("expert_ood", ood, None),
        ("bc_id", id_test, bc_params),
        ("bc_ood", ood, bc_params),
        ("dagger_id", id_test, dagger_params),
        ("dagger_ood", ood, dagger_params),
        ("mutant_shuffle", id_test, shuffle_params),
    ]:
        if fn is None:
            eps = []
            for w in worlds:
                ep = expert_episode(w, w.start, w.goal)
                ep["suite"] = name
                ep["boxes"] = w.boxes
                ep["invalid"] = ep["status"] == "invalid"
                eps.append(ep)
                episodes.append(ep)
            suites[name] = _aggregate(eps)
        elif name == "mutant_shuffle":
            agg, eps = eval_suite(worlds, fn, name)
            suites[name] = agg
            episodes.extend(eps)
        else:
            agg, eps = eval_suite(worlds, fn, name)
            suites[name] = agg
            episodes.extend(eps)

    zero_agg, zero_eps = eval_suite(id_test, bc_params, "mutant_zero", zero_policy_fn())
    suites["mutant_zero"] = zero_agg
    episodes.extend(zero_eps)

    episode_records = []
    for ep in episodes:
        rec = {k: v for k, v in ep.items() if k != "xy_traj"}
        if "invalid" not in rec:
            rec["invalid"] = rec["status"] == "invalid"
        episode_records.append(rec)

    for ep in episodes:
        if ep["suite"] in ("expert_id", "expert_ood", "bc_id", "bc_ood", "dagger_id", "dagger_ood"):
            xy = ep.get("xy_traj") or [[ep["start"][0], ep["start"][1]]]
            trajectories.append(
                {
                    "suite": ep["suite"],
                    "seed": ep["seed"],
                    "status": ep["status"],
                    "start": ep["start"],
                    "goal": ep["goal"],
                    "xy": subsample_trajectory([[p[0], p[1]] for p in xy]),
                    "obstacles": ep.get("boxes", []),
                    "border": 0.2,
                }
            )
        if ep["suite"] == "mutant_shuffle" and ep["seed"] == id_test[0].seed:
            xy = ep.get("xy_traj") or [[ep["start"][0], ep["start"][1]]]
            trajectories.append(
                {
                    "suite": ep["suite"],
                    "seed": ep["seed"],
                    "status": ep["status"],
                    "start": ep["start"],
                    "goal": ep["goal"],
                    "xy": subsample_trajectory([[p[0], p[1]] for p in xy]),
                    "obstacles": ep.get("boxes", []),
                    "border": 0.2,
                }
            )
    if zero_eps:
        ep = zero_eps[0]
        xy = ep.get("xy_traj") or [[ep["start"][0], ep["start"][1]]]
        trajectories.append(
            {
                "suite": "mutant_zero",
                "seed": ep["seed"],
                "status": ep["status"],
                "start": ep["start"],
                "goal": ep["goal"],
                "xy": subsample_trajectory([[p[0], p[1]] for p in xy]),
                "obstacles": ep.get("boxes", []),
                "border": 0.2,
            }
        )

    open_loop = {
        "bc_id": _open_loop_mse(bc_params, id_test),
        "dagger_id": _open_loop_mse(dagger_params, id_test),
        "shuffle_id": _open_loop_mse(shuffle_params, id_test),
    }

    report = build_report(
        git_sha if len(git_sha) == 40 else "0" * 40,
        run_id=str(uuid.uuid4()),
        suites=suites,
        episodes=episode_records,
        trajectories=trajectories,
        open_loop_mse=open_loop,
        dagger_selected_round=dagger_round,
    )
    write_report(out, report)
    return check_report(out, require_ci=False, expect_sha=None)


if __name__ == "__main__":
    main()
