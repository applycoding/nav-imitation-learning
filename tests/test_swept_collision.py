# SPDX-License-Identifier: Apache-2.0
"""Continuous clearance and swept contact tests."""

from __future__ import annotations

import math

from nav_imitation_learning.constants import DT, ROBOT_RADIUS_M
from nav_imitation_learning.env.collision import clearance, rectangles_from_boxes, swept_step


def test_clearance_contact_threshold() -> None:
    # Map center, 0.20 m from box face: clearance == 0 (grazing, not penetrating)
    box = [[5.0, 5.0, 1.0, 1.0]]
    rects = rectangles_from_boxes(box)
    pose = (5.0 - ROBOT_RADIUS_M, 5.5, 0.0)
    assert abs(clearance(pose, rects)) < 1e-9
    pose_in = (5.0 - 0.19, 5.5, 0.0)
    assert clearance(pose_in, rects) < 0.0


def test_grazing_corner() -> None:
    """Endpoints free along a chord; an intermediate arc sample penetrates the disk."""
    rects = rectangles_from_boxes([[5.0, 5.0, 0.5, 0.5]])
    # Straight segment from A to B passes closer than robot radius to the box corner
    start = (4.5, 5.25, 0.0)
    v = math.hypot(6.0 - 4.5, 5.25 - 5.25) / DT
    _, status = swept_step(start, v, 0.0, DT, rects)
    assert clearance(start, rects) >= 0.0
    assert status == "collision"


def test_start_already_penetrating() -> None:
    rects = rectangles_from_boxes([[1.0, 1.0, 1.0, 1.0]])
    pose = (1.2, 1.2, 0.0)
    end, status = swept_step(pose, 0.1, 0.0, DT, rects)
    assert status == "collision"
    assert end == pose
