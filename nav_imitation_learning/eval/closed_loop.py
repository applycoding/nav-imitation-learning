# SPDX-License-Identifier: Apache-2.0
"""Policy closed-loop rollouts."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from nav_imitation_learning.constants import DT, GOAL_TOL_XY, T_MAX_STEPS
from nav_imitation_learning.env.collision import clearance, plant_step
from nav_imitation_learning.env.maps import World
from nav_imitation_learning.env.plant import Command, Pose, pose_valid
from nav_imitation_learning.env.rays import observe
from nav_imitation_learning.policy.mlp import mlp_forward

EpisodeDict = dict[str, Any]
PolicyFn = Callable[[npt.NDArray[np.float32], Pose, Command, tuple[float, float]], tuple[float, float]]


def default_policy(
    params: dict[str, npt.NDArray[np.float64]],
) -> PolicyFn:
    def _fn(
        obs: npt.NDArray[np.float32],
        pose: Pose,
        prev: Command,
        goal: tuple[float, float],
    ) -> tuple[float, float]:
        del pose, prev, goal
        if not np.all(np.isfinite(obs)):
            return (float("nan"), float("nan"))
        _, cmd = mlp_forward(params, obs.astype(np.float64).reshape(1, -1))
        return float(cmd[0, 0]), float(cmd[0, 1])

    return _fn


def policy_episode(
    world: World,
    params: dict[str, npt.NDArray[np.float64]],
    start: Pose,
    goal: tuple[float, float],
    policy_fn: PolicyFn | None = None,
) -> EpisodeDict:
    if policy_fn is None:
        policy_fn = default_policy(params)
    return _run_episode(world, start, goal, policy_fn)


def _run_episode(
    world: World,
    start: Pose,
    goal: tuple[float, float],
    policy_fn: PolicyFn,
) -> EpisodeDict:
    pose = start
    prev: Command = (0.0, 0.0)
    steps = 0
    path_length = 0.0
    collided = False
    xy_traj: list[tuple[float, float]] = [(pose[0], pose[1])]

    if not pose_valid(pose[0], pose[1], pose[2]):
        return _record(world, start, goal, pose, steps, path_length, "invalid", False, xy_traj)
    if clearance(pose, world.rectangles) < 0.0:
        return _record(world, start, goal, pose, steps, path_length, "collision", True, xy_traj)
    if math.hypot(goal[0] - pose[0], goal[1] - pose[1]) <= GOAL_TOL_XY:
        return _record(world, start, goal, pose, steps, path_length, "success", False, xy_traj)

    while steps < T_MAX_STEPS:
        if not pose_valid(pose[0], pose[1], pose[2]):
            return _record(world, start, goal, pose, steps, path_length, "invalid", collided, xy_traj)
        if clearance(pose, world.rectangles) < 0.0:
            return _record(world, start, goal, pose, steps, path_length, "collision", True, xy_traj)
        if math.hypot(goal[0] - pose[0], goal[1] - pose[1]) <= GOAL_TOL_XY:
            return _record(world, start, goal, pose, steps, path_length, "success", collided, xy_traj)
        if steps >= T_MAX_STEPS:
            return _record(world, start, goal, pose, steps, path_length, "timeout", collided, xy_traj)

        obs = observe(world.occupancy, pose, goal, prev)
        if not np.all(np.isfinite(obs)):
            return _record(world, start, goal, pose, steps, path_length, "invalid", collided, xy_traj)
        v_req, w_req = policy_fn(obs, pose, prev, goal)
        if not math.isfinite(v_req) or not math.isfinite(w_req):
            return _record(world, start, goal, pose, steps, path_length, "invalid", collided, xy_traj)

        old_x, old_y = pose[0], pose[1]
        new_pose, new_prev, cst = plant_step(pose, v_req, w_req, prev, DT, world.rectangles)
        if cst == "invalid":
            return _record(world, start, goal, pose, steps, path_length, "invalid", collided, xy_traj)
        if cst == "collision":
            collided = True
            pose = new_pose
            xy_traj.append((pose[0], pose[1]))
            return _record(world, start, goal, pose, steps + 1, path_length, "collision", True, xy_traj)

        pose = new_pose
        prev = new_prev
        path_length += math.hypot(pose[0] - old_x, pose[1] - old_y)
        steps += 1
        xy_traj.append((pose[0], pose[1]))

        if math.hypot(goal[0] - pose[0], goal[1] - pose[1]) <= GOAL_TOL_XY:
            return _record(world, start, goal, pose, steps, path_length, "success", collided, xy_traj)

    return _record(world, start, goal, pose, steps, path_length, "timeout", collided, xy_traj)


def _record(
    world: World,
    start: Pose,
    goal: tuple[float, float],
    final: Pose,
    steps: int,
    path_length: float,
    status: str,
    collision: bool,
    xy_traj: list[tuple[float, float]],
) -> EpisodeDict:
    return {
        "suite": "",
        "seed": world.seed,
        "start": list(start),
        "goal": list(goal),
        "final_pose": list(final),
        "status": status,
        "collision": collision,
        "invalid": status == "invalid",
        "steps": steps,
        "time_s": steps * DT,
        "path_length_m": path_length,
        "final_err_m": math.hypot(goal[0] - final[0], goal[1] - final[1]),
        "xy_traj": xy_traj,
        "boxes": world.boxes,
    }


def eval_suite(
    worlds: list[World],
    params: dict[str, npt.NDArray[np.float64]],
    suite_name: str,
    policy_fn: PolicyFn | None = None,
) -> tuple[dict[str, Any], list[EpisodeDict]]:
    episodes: list[EpisodeDict] = []
    for world in worlds:
        ep = policy_episode(world, params, world.start, world.goal, policy_fn)
        ep["suite"] = suite_name
        episodes.append(ep)
    return _aggregate(episodes), episodes


def _aggregate(episodes: list[EpisodeDict]) -> dict[str, Any]:
    n = len(episodes)
    successes = sum(1 for e in episodes if e["status"] == "success")
    collisions = sum(1 for e in episodes if e["collision"])
    invalids = sum(1 for e in episodes if e["status"] == "invalid")
    timeouts = sum(1 for e in episodes if e["status"] == "timeout")
    succ_times = [e["time_s"] for e in episodes if e["status"] == "success"]
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
