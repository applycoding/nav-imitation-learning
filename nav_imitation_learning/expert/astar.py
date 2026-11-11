# SPDX-License-Identifier: Apache-2.0
"""8-connected A* on the planning mask."""

from __future__ import annotations

import heapq

import numpy as np
import numpy.typing as npt

from nav_imitation_learning.constants import MAP_RES_M, SQRT2

NEIGHBORS = [
    (1, 0, 1.0),
    (1, 1, SQRT2),
    (0, 1, 1.0),
    (-1, 1, SQRT2),
    (-1, 0, 1.0),
    (-1, -1, SQRT2),
    (0, -1, 1.0),
    (1, -1, SQRT2),
]


def octile_heuristic(i: int, j: int, gi: int, gj: int) -> float:
    dx = abs(gi - i)
    dy = abs(gj - j)
    return max(dx, dy) + (SQRT2 - 1.0) * min(dx, dy)


def astar(
    lethal: npt.NDArray[np.uint8],
    start_ij: tuple[int, int],
    goal_ij: tuple[int, int],
) -> list[tuple[int, int]] | None:
    if lethal[start_ij[0], start_ij[1]] or lethal[goal_ij[0], goal_ij[1]]:
        return None
    n_rows, n_cols = lethal.shape
    insert_counter = 0
    open_heap: list[tuple[float, float, int, int, int]] = []
    g_score: dict[tuple[int, int], float] = {start_ij: 0.0}
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    h0 = octile_heuristic(start_ij[0], start_ij[1], goal_ij[0], goal_ij[1])
    heapq.heappush(open_heap, (h0, 0.0, insert_counter, start_ij[0], start_ij[1]))
    insert_counter += 1
    closed: set[tuple[int, int]] = set()

    while open_heap:
        f, g, _ins, ci, cj = heapq.heappop(open_heap)
        if (ci, cj) in closed:
            continue
        if (ci, cj) == goal_ij:
            path: list[tuple[int, int]] = [(ci, cj)]
            while (ci, cj) in came_from:
                ci, cj = came_from[(ci, cj)]
                path.append((ci, cj))
            path.reverse()
            return path
        closed.add((ci, cj))
        for di, dj, cost in NEIGHBORS:
            ni, nj = ci + di, cj + dj
            if ni < 0 or nj < 0 or ni >= n_rows or nj >= n_cols:
                continue
            if lethal[ni, nj]:
                continue
            if abs(di) == 1 and abs(dj) == 1:
                if lethal[ci + di, cj] or lethal[ci, cj + dj]:
                    continue
            tentative = g + cost
            if tentative < g_score.get((ni, nj), float("inf")):
                g_score[(ni, nj)] = tentative
                came_from[(ni, nj)] = (ci, cj)
                h = octile_heuristic(ni, nj, goal_ij[0], goal_ij[1])
                f_new = tentative + h
                heapq.heappush(open_heap, (f_new, tentative, insert_counter, ni, nj))
                insert_counter += 1
    return None


def polyline_cell_centers(
    path: list[tuple[int, int]],
    start: tuple[float, float, float],
    goal: tuple[float, float],
    lethal: npt.NDArray[np.uint8],
) -> list[tuple[float, float]]:
    """Build polyline with continuous start and goal when segments stay in free space."""
    if not path:
        return []
    pts: list[tuple[float, float]] = []
    for _idx, (i, j) in enumerate(path):
        cx = (i + 0.5) * MAP_RES_M
        cy = (j + 0.5) * MAP_RES_M
        pts.append((cx, cy))
    if len(pts) >= 2:
        pts[0] = (start[0], start[1])
        if _segment_free(pts[0], pts[1], lethal):
            pass
        else:
            pts[0] = ((path[0][0] + 0.5) * MAP_RES_M, (path[0][1] + 0.5) * MAP_RES_M)
        pts[-1] = (goal[0], goal[1])
        if not _segment_free(pts[-2], pts[-1], lethal):
            pts[-1] = ((path[-1][0] + 0.5) * MAP_RES_M, (path[-1][1] + 0.5) * MAP_RES_M)
    return pts


def _segment_free(a: tuple[float, float], b: tuple[float, float], lethal: npt.NDArray[np.uint8]) -> bool:
    steps = int(max(abs(b[0] - a[0]), abs(b[1] - a[1])) / MAP_RES_M * 2) + 2
    for k in range(steps + 1):
        t = k / steps if steps else 0.0
        x = a[0] + t * (b[0] - a[0])
        y = a[1] + t * (b[1] - a[1])
        i = int(x / MAP_RES_M)
        j = int(y / MAP_RES_M)
        if i < 0 or j < 0 or i >= lethal.shape[0] or j >= lethal.shape[1]:
            return False
        if lethal[i, j]:
            return False
    return True
