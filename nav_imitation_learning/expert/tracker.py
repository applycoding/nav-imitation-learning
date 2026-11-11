# SPDX-License-Identifier: Apache-2.0
"""Lookahead heading controller along an A* polyline."""

from __future__ import annotations

import math
from dataclasses import dataclass

from nav_imitation_learning.constants import GOAL_TOL_XY, HEADING_KP, LOOKAHEAD_M, V_CRUISE
from nav_imitation_learning.env.plant import Pose, wrap


@dataclass
class TrackerProgress:
    arc_length: float = 0.0


def project_arc_on_polyline(polyline: list[tuple[float, float]], pose: Pose) -> float:
    """Arc length of the nearest point on the polyline (stateless projection)."""
    x, y, _ = pose
    if len(polyline) < 2:
        return 0.0
    seg_lens = [
        math.hypot(polyline[i + 1][0] - polyline[i][0], polyline[i + 1][1] - polyline[i][1])
        for i in range(len(polyline) - 1)
    ]
    cumulative = [0.0]
    for sl in seg_lens:
        cumulative.append(cumulative[-1] + sl)
    best_dist = float("inf")
    best_arc = 0.0
    for i in range(len(polyline) - 1):
        ax, ay = polyline[i]
        bx, by = polyline[i + 1]
        dx, dy = bx - ax, by - ay
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < 1e-18:
            t_proj = 0.0
        else:
            t_proj = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / seg_len_sq))
        px = ax + t_proj * dx
        py = ay + t_proj * dy
        dist = math.hypot(x - px, y - py)
        arc = cumulative[i] + t_proj * math.sqrt(seg_len_sq)
        if dist < best_dist - 1e-9:
            best_dist = dist
            best_arc = arc
        elif abs(dist - best_dist) < 1e-9:
            best_arc = min(best_arc, arc)
    return best_arc


def tracker_raw(
    pose: Pose,
    polyline: list[tuple[float, float]],
    progress: TrackerProgress,
    goal: tuple[float, float],
) -> tuple[float, float, TrackerProgress]:
    x, y, theta = pose
    gx, gy = goal
    if math.hypot(gx - x, gy - y) <= GOAL_TOL_XY:
        return 0.0, 0.0, progress
    if len(polyline) < 2:
        e = wrap(math.atan2(gy - y, gx - x) - theta)
        return 0.0, HEADING_KP * e, progress

    _seg_lens = [
        math.hypot(polyline[i + 1][0] - polyline[i][0], polyline[i + 1][1] - polyline[i][1])
        for i in range(len(polyline) - 1)
    ]
    cumulative = [0.0]
    for sl in _seg_lens:
        cumulative.append(cumulative[-1] + sl)

    best_dist = float("inf")
    best_arc = progress.arc_length
    for i in range(len(polyline) - 1):
        ax, ay = polyline[i]
        bx, by = polyline[i + 1]
        dx, dy = bx - ax, by - ay
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < 1e-18:
            t_proj = 0.0
        else:
            t_proj = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / seg_len_sq))
        px = ax + t_proj * dx
        py = ay + t_proj * dy
        dist = math.hypot(x - px, y - py)
        arc = cumulative[i] + t_proj * math.sqrt(seg_len_sq)
        if dist < best_dist - 1e-9:
            best_dist = dist
            best_arc = arc
        elif abs(dist - best_dist) < 1e-9:
            best_arc = min(best_arc, arc)

    arc = max(progress.arc_length, best_arc)
    progress = TrackerProgress(arc_length=arc)

    target_arc = arc + LOOKAHEAD_M
    lx, ly = _point_at_arc(polyline, cumulative, target_arc)
    e = wrap(math.atan2(ly - y, lx - x) - theta)
    dist_goal = math.hypot(gx - x, gy - y)

    if abs(e) > math.pi / 2:
        v_raw = 0.0
        w_raw = HEADING_KP * e
    elif dist_goal < LOOKAHEAD_M:
        v_raw = V_CRUISE * dist_goal / LOOKAHEAD_M
        w_raw = HEADING_KP * e
    else:
        v_raw = V_CRUISE
        w_raw = HEADING_KP * e
    return v_raw, w_raw, progress


def _point_at_arc(
    poly: list[tuple[float, float]],
    cumulative: list[float],
    target: float,
) -> tuple[float, float]:
    if target >= cumulative[-1]:
        return poly[-1]
    for i in range(len(cumulative) - 1):
        if cumulative[i + 1] >= target:
            local = (target - cumulative[i]) / (cumulative[i + 1] - cumulative[i] + 1e-18)
            ax, ay = poly[i]
            bx, by = poly[i + 1]
            return (ax + local * (bx - ax), ay + local * (by - ay))
    return poly[-1]
