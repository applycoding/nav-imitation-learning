# SPDX-License-Identifier: Apache-2.0
"""Expert demonstration dataset collection."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from nav_imitation_learning.constants import (
    DT,
    GOAL_TOL_XY,
    MAP_RES_M,
    OMEGA_MAX,
    T_MAX_STEPS,
    V_MAX,
)
from nav_imitation_learning.env.collision import clearance
from nav_imitation_learning.env.maps import World
from nav_imitation_learning.env.plant import Command, Pose, apply_command, pose_valid
from nav_imitation_learning.env.rays import observe
from nav_imitation_learning.expert.astar import astar, polyline_cell_centers
from nav_imitation_learning.expert.rollout import _expert_safety_step
from nav_imitation_learning.expert.tracker import TrackerProgress, tracker_raw


def collect_expert_demonstrations(world: World) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    obs_rows: list[npt.NDArray[np.float64]] = []
    act_rows: list[npt.NDArray[np.float64]] = []
    pose: Pose = world.start
    prev: Command = (0.0, 0.0)
    goal = world.goal

    if not pose_valid(pose[0], pose[1], pose[2]) or clearance(pose, world.rectangles) < 0.0:
        return np.zeros((0, 40)), np.zeros((0, 2))
    si = int(pose[0] / MAP_RES_M)
    sj = int(pose[1] / MAP_RES_M)
    gi = int(goal[0] / MAP_RES_M)
    gj = int(goal[1] / MAP_RES_M)
    path = astar(world.lethal, (si, sj), (gi, gj))
    if path is None:
        return np.zeros((0, 40)), np.zeros((0, 2))
    poly = polyline_cell_centers(path, world.start, goal, world.lethal)
    progress = TrackerProgress(0.0)

    for _ in range(T_MAX_STEPS):
        if math.hypot(goal[0] - pose[0], goal[1] - pose[1]) <= GOAL_TOL_XY:
            break
        if clearance(pose, world.rectangles) < 0.0:
            break

        o = observe(world.occupancy, pose, goal, prev)
        v_raw, w_raw, progress = tracker_raw(pose, poly, progress, goal)
        v_req, w_req, st = apply_command(v_raw, w_raw, prev, DT)
        if st == "invalid":
            break

        new_pose, new_prev, cst = _expert_safety_step(
            pose, v_req, w_req, prev, poly, progress, goal, world
        )
        if cst != "ok":
            break

        obs_rows.append(o.astype(np.float64))
        act_rows.append(np.array([new_prev[0] / V_MAX, new_prev[1] / OMEGA_MAX], dtype=np.float64))
        pose = new_pose
        prev = new_prev

    if not obs_rows:
        return np.zeros((0, 40)), np.zeros((0, 2))
    return np.stack(obs_rows), np.stack(act_rows)


def collect_all_train_demos(train_worlds: list[World]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    obs_all: list[npt.NDArray[np.float64]] = []
    act_all: list[npt.NDArray[np.float64]] = []
    for w in train_worlds:
        o, a = collect_expert_demonstrations(w)
        if o.size:
            obs_all.append(o)
            act_all.append(a)
    if not obs_all:
        return np.zeros((0, 40)), np.zeros((0, 2))
    return np.vstack(obs_all), np.vstack(act_all)
