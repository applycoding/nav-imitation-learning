# SPDX-License-Identifier: Apache-2.0
"""Amanatides-Woo raycast and observation packing tests."""

from __future__ import annotations

import math

import numpy as np

from nav_imitation_learning.constants import MAP_CELLS, MAP_RES_M, R_MAX
from nav_imitation_learning.env.plant import pose_valid
from nav_imitation_learning.env.rays import pack_obs, ray_ranges


def _grid_with_box() -> np.ndarray:
    occ = np.zeros((MAP_CELLS, MAP_CELLS), dtype=np.uint8)
    # Occupied rectangle [2.0, 3.0] x [0.0, 2.0]
    i0 = int(2.0 / MAP_RES_M)
    i1 = int(3.0 / MAP_RES_M)
    j0 = int(0.0 / MAP_RES_M)
    j1 = int(2.0 / MAP_RES_M)
    occ[i0:i1, j0:j1] = 1
    return occ


def test_box_edge() -> None:
    occ = _grid_with_box()
    ranges = ray_ranges(occ, 1.0, 1.0, 0.0, MAP_RES_M)
    assert abs(ranges[0] - 1.0) < 1e-9
    obs = pack_obs(ranges, (9.0, 9.0), (1.0, 1.0, 0.0), (0.0, 0.0))
    assert abs(obs[0] - 0.25) < 1e-9


def test_start_inside_occupied_zero_range() -> None:
    occ = _grid_with_box()
    ranges = ray_ranges(occ, 2.25, 1.0, 0.0, MAP_RES_M)
    assert ranges[0] == 0.0


def test_all_max_range_ones() -> None:
    occ = np.zeros((MAP_CELLS, MAP_CELLS), dtype=np.uint8)
    ranges = ray_ranges(occ, 5.0, 5.0, 0.0, MAP_RES_M)
    clipped = np.minimum(ranges, R_MAX) / R_MAX
    assert np.allclose(clipped, 1.0)


def test_invalid_pose_nan_obs() -> None:
    occ = np.zeros((MAP_CELLS, MAP_CELLS), dtype=np.uint8)
    obs = pack_obs(
        ray_ranges(occ, 11.0, 5.0, 0.0, MAP_RES_M),
        (9.0, 9.0),
        (11.0, 5.0, 0.0),
        (0.0, 0.0),
    )
    assert np.all(np.isnan(obs))
    assert not pose_valid(11.0, 5.0, 0.0)


def test_goal_frame_rotation() -> None:
    np.zeros((MAP_CELLS, MAP_CELLS), dtype=np.uint8)
    ranges = np.full(36, R_MAX, dtype=np.float64)
    obs = pack_obs(ranges, (5.0, 6.0), (5.0, 5.0, math.pi / 2), (0.3, 0.0))
    assert abs(obs[36] - 1.0 / R_MAX) < 1e-6
    assert abs(obs[37]) < 1e-6
