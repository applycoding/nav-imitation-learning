# SPDX-License-Identifier: Apache-2.0
"""Expert closed-loop rollout."""

from __future__ import annotations

import math
from typing import Any

from nav_imitation_learning.constants import DT, GOAL_TOL_XY, MAP_RES_M, T_MAX_STEPS
from nav_imitation_learning.env.collision import clearance, plant_step
from nav_imitation_learning.env.maps import World
from nav_imitation_learning.env.plant import Command, Pose, apply_command, pose_valid
from nav_imitation_learning.expert.astar import astar, polyline_cell_centers
from nav_imitation_learning.expert.tracker import TrackerProgress, tracker_raw

EpisodeDict = dict[str, Any]


def expert_episode(world: World, start: Pose, goal: tuple[float, float]) -> EpisodeDict:
    """Run the scripted expert; the record carries `xy_traj` (every plant pose) for drawing."""
    xy: list[list[float]] = [[start[0], start[1]]]
    rec = _expert_episode(world, start, goal, xy)
    rec["xy_traj"] = xy
    return rec


def _expert_episode(
    world: World,
    start: Pose,
    goal: tuple[float, float],
    xy: list[list[float]],
) -> EpisodeDict:
    pose = start
    prev: Command = (0.0, 0.0)
    steps = 0
    path_length = 0.0
    collided = False

    if not pose_valid(pose[0], pose[1], pose[2]):
        return _episode_record(world, start, goal, pose, steps, path_length, "invalid", True, False)

    if clearance(pose, world.rectangles) < 0.0:
        return _episode_record(world, start, goal, pose, steps, path_length, "collision", True, False)

    if math.hypot(goal[0] - pose[0], goal[1] - pose[1]) <= GOAL_TOL_XY:
        return _episode_record(world, start, goal, pose, steps, path_length, "success", False, False)

    si = int(pose[0] / MAP_RES_M)
    sj = int(pose[1] / MAP_RES_M)
    gi = int(goal[0] / MAP_RES_M)
    gj = int(goal[1] / MAP_RES_M)
    path = astar(world.lethal, (si, sj), (gi, gj))
    if path is None:
        return _episode_record(world, start, goal, pose, steps, path_length, "invalid", False, False)

    poly = polyline_cell_centers(path, start, goal, world.lethal)
    progress = TrackerProgress(0.0)

    while steps < T_MAX_STEPS:
        if not pose_valid(pose[0], pose[1], pose[2]):
            return _episode_record(world, start, goal, pose, steps, path_length, "invalid", collided, False)
        if clearance(pose, world.rectangles) < 0.0:
            return _episode_record(world, start, goal, pose, steps, path_length, "collision", True, False)
        if math.hypot(goal[0] - pose[0], goal[1] - pose[1]) <= GOAL_TOL_XY:
            return _episode_record(world, start, goal, pose, steps, path_length, "success", collided, False)

        v_raw, w_raw, progress = tracker_raw(pose, poly, progress, goal)
        v_app, w_app, st = apply_command(v_raw, w_raw, prev, DT)
        if st == "invalid":
            return _episode_record(world, start, goal, pose, steps, path_length, "invalid", collided, False)

        new_pose, new_prev, cst = _expert_safety_step(pose, v_app, w_app, prev, poly, progress, goal, world)
        if cst == "invalid":
            return _episode_record(world, start, goal, pose, steps, path_length, "invalid", collided, False)
        if cst == "collision":
            xy.append([new_pose[0], new_pose[1]])
            return _episode_record(world, start, goal, new_pose, steps, path_length, "collision", True, False)

        old_x, old_y = pose[0], pose[1]
        pose = new_pose
        prev = new_prev
        path_length += math.hypot(pose[0] - old_x, pose[1] - old_y)
        steps += 1
        xy.append([pose[0], pose[1]])

        if math.hypot(goal[0] - pose[0], goal[1] - pose[1]) <= GOAL_TOL_XY:
            return _episode_record(world, start, goal, pose, steps, path_length, "success", collided, False)

    return _episode_record(world, start, goal, pose, steps, path_length, "timeout", collided, False)


def _expert_safety_step(
    pose: Pose,
    v_app: float,
    w_app: float,
    prev: Command,
    poly: list[tuple[float, float]],
    progress: TrackerProgress,
    goal: tuple[float, float],
    world: World,
) -> tuple[Pose, Command, str]:
    new_pose, new_prev, cst = plant_step(pose, v_app, w_app, prev, DT, world.rectangles)
    if cst != "collision":
        return new_pose, new_prev, cst
    v2, w2, _ = tracker_raw(pose, poly, progress, goal)
    e = w2 / 1.5 if abs(1.5) > 1e-12 else 0.0
    va, wb, _ = apply_command(0.0, 1.5 * e, prev, DT)
    new_pose, new_prev, cst2 = plant_step(pose, va, wb, prev, DT, world.rectangles)
    if cst2 != "collision":
        return new_pose, new_prev, cst2
    va3, wb3, _ = apply_command(0.0, 0.0, prev, DT)
    new_pose, new_prev, cst3 = plant_step(pose, va3, wb3, prev, DT, world.rectangles)
    return new_pose, new_prev, cst3


def _episode_record(
    world: World,
    start: Pose,
    goal: tuple[float, float],
    final: Pose,
    steps: int,
    path_length: float,
    status: str,
    collision: bool,
    timeout_flag: bool,
) -> EpisodeDict:
    del timeout_flag
    return {
        "seed": world.seed,
        "start": list(start),
        "goal": list(goal),
        "final_pose": list(final),
        "status": status,
        "collision": collision,
        "steps": steps,
        "time_s": steps * DT,
        "path_length_m": path_length,
        "final_err_m": math.hypot(goal[0] - final[0], goal[1] - final[1]),
    }
