# SPDX-License-Identifier: Apache-2.0
"""A* planner unit tests."""

from __future__ import annotations

import numpy as np

from nav_imitation_learning.constants import SQRT2
from nav_imitation_learning.expert.astar import astar, octile_heuristic


def test_octile_empty_grid() -> None:
    lethal = np.zeros((10, 10), dtype=np.uint8)
    h = octile_heuristic(0, 0, 3, 1)
    assert abs(h - (SQRT2 + 2.0)) < 1e-12
    path = astar(lethal, (0, 0), (3, 1))
    assert path is not None
    cost = 0.0
    for i in range(len(path) - 1):
        di = path[i + 1][0] - path[i][0]
        dj = path[i + 1][1] - path[i][1]
        step = SQRT2 if di != 0 and dj != 0 else 1.0
        cost += step
    assert abs(cost - (SQRT2 + 2.0)) < 1e-9


def test_no_corner_cut() -> None:
    lethal = np.zeros((5, 5), dtype=np.uint8)
    # Notch: block (1,1) diagonal corner while orthogonal neighbors free for start
    lethal[1, 1] = 1
    lethal[0, 1] = 0
    lethal[1, 0] = 0
    path = astar(lethal, (0, 0), (2, 2))
    assert path is not None
    assert (1, 1) not in path
    for a, b in zip(path, path[1:], strict=False):
        di = abs(b[0] - a[0])
        dj = abs(b[1] - a[1])
        if di == 1 and dj == 1:
            orth1 = (a[0] + (b[0] - a[0]), a[1])
            orth2 = (a[0], a[1] + (b[1] - a[1]))
            assert not lethal[orth1[0], orth1[1]]
            assert not lethal[orth2[0], orth2[1]]


def test_lethal_start_none() -> None:
    lethal = np.ones((4, 4), dtype=np.uint8)
    lethal[0, 0] = 0
    assert astar(lethal, (1, 1), (0, 0)) is None
