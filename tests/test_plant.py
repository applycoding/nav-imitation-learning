# SPDX-License-Identifier: Apache-2.0
"""Unicycle integration, wrapping, and command pipeline tests."""

from __future__ import annotations

import math

from nav_imitation_learning.constants import DT
from nav_imitation_learning.env.plant import apply_command, step_unicycle, wrap


def test_quarter_circle() -> None:
    pi2 = math.pi / 2
    x, y, th = step_unicycle((0.0, 0.0, 0.0), 1.0, 1.0, pi2)
    assert abs(x - 1.0) < 1e-12
    assert abs(y - 1.0) < 1e-12
    assert abs(th - pi2) < 1e-12


def test_small_angle() -> None:
    dtheta = 0.06
    s = math.sin(0.03) / 0.03
    expect_dx = 0.5 * 0.05 * s * math.cos(0.03)
    expect_dy = 0.5 * 0.05 * s * math.sin(0.03)
    x, y, _ = step_unicycle((0.0, 0.0, 0.0), 0.5, 1.2, 0.05)
    assert abs(x - expect_dx) < 1e-12
    assert abs(y - expect_dy) < 1e-12
    assert abs(dtheta - 0.06) < 1e-12


def test_straight_step() -> None:
  x, y, th = step_unicycle((0.0, 0.0, 0.0), 0.5, 0.0, 0.05)
  assert abs(x - 0.025) < 1e-12
  assert abs(y) < 1e-12
  assert abs(th) < 1e-12


def test_wrap_pi() -> None:
  assert wrap(-math.pi) == math.pi
  assert wrap(math.pi) == math.pi


def test_rate_limit_first_step() -> None:
  v, w, st = apply_command(0.60, 1.20, (0.0, 0.0), DT)
  assert st == "ok"
  assert abs(v - 0.05) < 1e-12
  assert abs(w - 0.15) < 1e-12


def test_nan_command_invalid() -> None:
  v, w, st = apply_command(float("nan"), 0.0, (0.0, 0.0), DT)
  assert st == "invalid"
  assert math.isnan(v)


def test_non_positive_dt_invalid() -> None:
  out = step_unicycle((0.0, 0.0, 0.0), 0.1, 0.0, 0.0)
  assert all(math.isnan(c) for c in out)
