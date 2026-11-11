# SPDX-License-Identifier: Apache-2.0
"""DAgger-style aggregation training."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import numpy.typing as npt

from nav_imitation_learning.constants import DAGGER_EPOCHS, DAGGER_ROUNDS, OMEGA_MAX, V_MAX
from nav_imitation_learning.env.maps import World
from nav_imitation_learning.eval.closed_loop import policy_episode
from nav_imitation_learning.expert.label import label
from nav_imitation_learning.train.bc import train_bc


def train_dagger(
    init_params: dict[str, npt.NDArray[np.float64]],
    train_worlds: list[World],
    obs: npt.NDArray[np.float64],
    actions: npt.NDArray[np.float64],
) -> tuple[dict[str, npt.NDArray[np.float64]], dict[str, Any]]:
    xavier = {k: v.copy() for k, v in init_params.items()}
    aggregated_obs = [obs]
    aggregated_act = [actions]
    checkpoints: list[tuple[int, dict[str, npt.NDArray[np.float64]]]] = []

    bc_params = train_bc(copy.deepcopy(xavier), obs, actions)
    checkpoints.append((0, {k: v.copy() for k, v in bc_params.items()}))

    current = bc_params
    for round_idx in range(1, DAGGER_ROUNDS + 1):
        round_obs: list[npt.NDArray[np.float64]] = []
        round_act: list[npt.NDArray[np.float64]] = []
        for world in train_worlds:
            rows = _dagger_rollout_rows(world, current)
            if rows[0].size:
                round_obs.append(rows[0])
                round_act.append(rows[1])
        if round_obs:
            aggregated_obs.append(np.vstack(round_obs))
            aggregated_act.append(np.vstack(round_act))
        agg_o = np.vstack(aggregated_obs)
        agg_a = np.vstack(aggregated_act)
        current = train_bc(copy.deepcopy(xavier), agg_o, agg_a, epochs=DAGGER_EPOCHS)
        checkpoints.append((round_idx, {k: v.copy() for k, v in current.items()}))

    train_scores: list[tuple[int, int, int, dict[str, npt.NDArray[np.float64]]]] = []
    for rnd, params in checkpoints:
        succ = 0
        coll = 0
        for world in train_worlds:
            ep = policy_episode(world, params, world.start, world.goal)
            if ep["status"] == "success":
                succ += 1
            if ep["collision"]:
                coll += 1
        train_scores.append((succ, -coll, rnd, params))

    train_scores.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
    best = train_scores[0]
    meta = {"dagger_selected_round": best[2], "train_success": best[0], "train_collisions": -best[1]}
    return best[3], meta


def _dagger_rollout_rows(
    world: World,
    params: dict[str, npt.NDArray[np.float64]],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    import math

    from nav_imitation_learning.constants import DT, GOAL_TOL_XY, T_MAX_STEPS
    from nav_imitation_learning.env.collision import clearance, plant_step
    from nav_imitation_learning.env.plant import Command, Pose
    from nav_imitation_learning.env.rays import observe
    from nav_imitation_learning.policy.mlp import mlp_forward

    obs_rows: list[npt.NDArray[np.float64]] = []
    act_rows: list[npt.NDArray[np.float64]] = []
    pose: Pose = world.start
    prev: Command = (0.0, 0.0)
    goal = world.goal
    for _ in range(T_MAX_STEPS):
        if math.hypot(goal[0] - pose[0], goal[1] - pose[1]) <= GOAL_TOL_XY:
            break
        if clearance(pose, world.rectangles) < 0.0:
            break
        o = observe(world.occupancy, pose, goal, prev)
        if not np.all(np.isfinite(o)):
            break
        lab = label(world, pose, prev, goal)
        if lab[0] == "ok":
            obs_rows.append(o.astype(np.float64))
            act_rows.append(np.array([lab[1] / V_MAX, lab[2] / OMEGA_MAX], dtype=np.float64))
        _, cmd = mlp_forward(params, o.astype(np.float64).reshape(1, -1))
        v_req, w_req = float(cmd[0, 0]), float(cmd[0, 1])
        new_pose, new_prev, status = plant_step(pose, v_req, w_req, prev, DT, world.rectangles)
        if status != "ok":
            break
        pose, prev = new_pose, new_prev
    if not obs_rows:
        return np.zeros((0, 40)), np.zeros((0, 2))
    return np.stack(obs_rows), np.stack(act_rows)
