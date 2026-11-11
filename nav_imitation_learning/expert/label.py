# SPDX-License-Identifier: Apache-2.0
"""Stateless expert label query for demonstrations and DAgger."""

from __future__ import annotations

from typing import Literal

from nav_imitation_learning.constants import DT, MAP_RES_M
from nav_imitation_learning.env.collision import clearance, plant_step
from nav_imitation_learning.env.maps import World
from nav_imitation_learning.env.plant import Command, Pose, apply_command, pose_valid
from nav_imitation_learning.expert.astar import astar, polyline_cell_centers
from nav_imitation_learning.expert.tracker import (
    TrackerProgress,
    project_arc_on_polyline,
    tracker_raw,
)

LabelResult = tuple[Literal["ok"], float, float] | tuple[Literal["unlabelable"], str]

_POLY_CACHE: dict[int, list[tuple[float, float]]] = {}


def _episode_polyline(world: World, goal: tuple[float, float]) -> list[tuple[float, float]] | None:
    key = world.seed
    if key in _POLY_CACHE:
        return _POLY_CACHE[key]
    si0 = int(world.start[0] / MAP_RES_M)
    sj0 = int(world.start[1] / MAP_RES_M)
    gi = int(goal[0] / MAP_RES_M)
    gj = int(goal[1] / MAP_RES_M)
    path = astar(world.lethal, (si0, sj0), (gi, gj))
    if path is None:
        return None
    poly = polyline_cell_centers(path, world.start, goal, world.lethal)
    _POLY_CACHE[key] = poly
    return poly


def label(
    world: World,
    pose: Pose,
    prev_applied: Command,
    goal: tuple[float, float],
) -> LabelResult:
    if not pose_valid(pose[0], pose[1], pose[2]):
        return ("unlabelable", "invalid_pose")
    if clearance(pose, world.rectangles) < 0.0:
        return ("unlabelable", "in_contact")
    poly = _episode_polyline(world, goal)
    if poly is None:
        return ("unlabelable", "no_path")
    progress = TrackerProgress(arc_length=project_arc_on_polyline(poly, pose))
    v_raw, w_raw, _ = tracker_raw(pose, poly, progress, goal)
    v_app, w_app, st = apply_command(v_raw, w_raw, prev_applied, DT)
    if st == "invalid":
        return ("unlabelable", "command_invalid")
    _, new_prev, cst = plant_step(pose, v_app, w_app, prev_applied, DT, world.rectangles)
    if cst == "invalid":
        return ("unlabelable", "command_invalid")
    if cst == "collision":
        v_app2, w_app2, _ = apply_command(0.0, w_raw, prev_applied, DT)
        _, new_prev2, cst2 = plant_step(pose, v_app2, w_app2, prev_applied, DT, world.rectangles)
        if cst2 == "invalid":
            return ("unlabelable", "command_invalid")
        if cst2 == "collision":
            v_app3, w_app3, _ = apply_command(0.0, 0.0, prev_applied, DT)
            _, new_prev3, cst3 = plant_step(pose, v_app3, w_app3, prev_applied, DT, world.rectangles)
            if cst3 == "invalid":
                return ("unlabelable", "command_invalid")
            if cst3 == "collision":
                return ("unlabelable", "safety_brake_collision")
            return ("ok", new_prev3[0], new_prev3[1])
        return ("ok", new_prev2[0], new_prev2[1])
    return ("ok", new_prev[0], new_prev[1])
