# SPDX-License-Identifier: Apache-2.0
"""Amanatides-Woo raycast and 40-D observations."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from nav_imitation_learning.constants import MAP_CELLS, MAP_RES_M, N_RAYS, OMEGA_MAX, R_MAX, V_MAX
from nav_imitation_learning.env.plant import Command, Pose, pose_valid

INF = float("inf")


def ray_ranges(
    occupancy: npt.NDArray[np.uint8],
    x: float,
    y: float,
    theta: float,
    res_m: float,
) -> npt.NDArray[np.float64]:
    out = np.zeros(N_RAYS, dtype=np.float64)
    if not pose_valid(x, y, theta):
        out[:] = float("nan")
        return out
    for k in range(N_RAYS):
        beam = -math.pi + k * (2.0 * math.pi / N_RAYS)
        world_angle = theta + beam
        out[k] = _ray_one(occupancy, x, y, world_angle, res_m)
    return out


def _ray_one(
    occ: npt.NDArray[np.uint8],
    x: float,
    y: float,
    angle: float,
    res_m: float,
) -> float:
    dx = math.cos(angle)
    dy = math.sin(angle)
    i = int(math.floor(x / res_m))
    j = int(math.floor(y / res_m))
    if 0 <= i < MAP_CELLS and 0 <= j < MAP_CELLS and occ[i, j]:
        return 0.0

    if abs(dx) < 1e-15:
        step_x = 0
        t_delta_x = INF
        t_max_x = INF
    else:
        step_x = 1 if dx > 0 else -1
        if dx > 0:
            next_boundary_x = (math.floor(x / res_m) + 1) * res_m
        else:
            next_boundary_x = math.floor(x / res_m) * res_m
        t_max_x = (next_boundary_x - x) / dx
        t_delta_x = abs(res_m / dx)

    if abs(dy) < 1e-15:
        step_y = 0
        t_delta_y = INF
        t_max_y = INF
    else:
        step_y = 1 if dy > 0 else -1
        if dy > 0:
            next_boundary_y = (math.floor(y / res_m) + 1) * res_m
        else:
            next_boundary_y = math.floor(y / res_m) * res_m
        t_max_y = (next_boundary_y - y) / dy
        t_delta_y = abs(res_m / dy)

    for _ in range(MAP_CELLS * 2 + 2):
        if t_max_x < t_max_y:
            t_hit = t_max_x
            t_max_x += t_delta_x
            i += step_x
        else:
            t_hit = t_max_y
            t_max_y += t_delta_y
            j += step_y

        if i < 0 or j < 0 or i >= MAP_CELLS or j >= MAP_CELLS:
            return max(0.0, t_hit)
        if occ[i, j]:
            return max(0.0, t_hit)
    return R_MAX


def pack_obs(
    ranges_m: npt.NDArray[np.float64],
    goal_xy: tuple[float, float],
    pose: Pose,
    prev_cmd: Command,
) -> npt.NDArray[np.float32]:
    x, y, theta = pose
    if not pose_valid(x, y, theta) or not np.all(np.isfinite(ranges_m)):
        return np.full(40, np.nan, dtype=np.float32)
    obs = np.zeros(40, dtype=np.float32)
    clipped = np.minimum(ranges_m, R_MAX) / R_MAX
    obs[:36] = clipped.astype(np.float32)
    gx = goal_xy[0] - x
    gy = goal_xy[1] - y
    c, s = math.cos(theta), math.sin(theta)
    grx = c * gx + s * gy
    gry = -s * gx + c * gy
    obs[36] = float(max(-R_MAX, min(R_MAX, grx)) / R_MAX)
    obs[37] = float(max(-R_MAX, min(R_MAX, gry)) / R_MAX)
    obs[38] = float(prev_cmd[0] / V_MAX)
    obs[39] = float(prev_cmd[1] / OMEGA_MAX)
    return obs


def observe(
    world_occupancy: npt.NDArray[np.uint8],
    pose: Pose,
    goal: tuple[float, float],
    prev: Command,
) -> npt.NDArray[np.float32]:
    ranges = ray_ranges(world_occupancy, pose[0], pose[1], pose[2], MAP_RES_M)
    return pack_obs(ranges, goal, pose, prev)
