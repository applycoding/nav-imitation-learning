# SPDX-License-Identifier: Apache-2.0
"""Continuous disk-to-rectangle clearance and swept contact checks."""

from __future__ import annotations

import math
from typing import Literal

from nav_imitation_learning.constants import MAP_SIZE_M, ROBOT_RADIUS_M
from nav_imitation_learning.env.plant import Command, Pose, apply_command, pose_valid, step_unicycle

Rect = tuple[float, float, float, float]
CollisionStatus = Literal["ok", "collision", "invalid"]

_OBSTACLE_RECTS: list[Rect] | None = None
_BORDER_RECTS: list[Rect] = [
    (0.0, 0.0, MAP_SIZE_M, 0.20),
    (0.0, MAP_SIZE_M - 0.20, MAP_SIZE_M, 0.20),
    (0.0, 0.0, 0.20, MAP_SIZE_M),
    (MAP_SIZE_M - 0.20, 0.0, 0.20, MAP_SIZE_M),
]


def _point_aabb_distance(px: float, py: float, rect: Rect) -> float:
    x, y, w, h = rect
    cx = max(x, min(px, x + w))
    cy = max(y, min(py, y + h))
    return math.hypot(px - cx, py - cy)


def clearance(pose: Pose, rectangles: list[Rect]) -> float:
    """Euclidean distance from disk center to union of rectangles, minus radius."""
    x, y, _ = pose
    if not pose_valid(x, y, pose[2]):
        return float("nan")
    all_rects = list(rectangles) + _BORDER_RECTS
    if not all_rects:
        return float("inf") - ROBOT_RADIUS_M
    min_dist = min(_point_aabb_distance(x, y, r) for r in all_rects)
    return min_dist - ROBOT_RADIUS_M


def rectangles_from_boxes(boxes: list[list[float]]) -> list[Rect]:
    return [(b[0], b[1], b[2], b[3]) for b in boxes]


def swept_step(
    state: Pose,
    v_applied: float,
    w_applied: float,
    dt: float,
    rectangles: list[Rect],
) -> tuple[Pose, CollisionStatus]:
    """Advance the plant with swept intermediate contact samples."""
    x, y, theta = state
    if not pose_valid(x, y, theta):
        return state, "invalid"
    if not math.isfinite(v_applied) or not math.isfinite(w_applied) or dt <= 0.0:
        return state, "invalid"

    start_clr = clearance(state, rectangles)
    if start_clr < 0.0:
        return state, "collision"

    k = max(2, int(math.ceil(abs(v_applied) * dt / 0.01)))
    last_in_world: Pose = state
    for step_i in range(1, k + 1):
        frac = step_i / k
        partial_dt = dt * frac
        sample = step_unicycle(state, v_applied, w_applied, partial_dt)
        sx, sy, st = sample
        if not pose_valid(sx, sy, st):
            return last_in_world, "invalid"
        clr = clearance(sample, rectangles)
        if clr < 0.0:
            return last_in_world, "collision"
        last_in_world = sample

    return last_in_world, "ok"


def plant_step(
    state: Pose,
    v_raw: float,
    w_raw: float,
    prev_cmd: Command,
    dt: float,
    rectangles: list[Rect],
) -> tuple[Pose, Command, CollisionStatus]:
    """apply_command once then swept_step."""
    v_app, w_app, st = apply_command(v_raw, w_raw, prev_cmd, dt)
    if st == "invalid":
        return state, prev_cmd, "invalid"
    new_pose, cst = swept_step(state, v_app, w_app, dt, rectangles)
    return new_pose, (v_app, w_app), cst
