# SPDX-License-Identifier: Apache-2.0
"""Unicycle kinematics, command rate limits, and angle wrapping."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import numpy.typing as npt

from nav_imitation_learning.constants import A_MAX, ALPHA_MAX, OMEGA_MAX, V_MAX

Pose = tuple[float, float, float]
Command = tuple[float, float]
StepStatus = Literal["ok", "invalid"]


def wrap(theta: float) -> float:
    """Map angle to (-pi, pi], with wrap(-pi) = pi."""
    if not math.isfinite(theta):
        return theta
    t = math.fmod(theta + math.pi, 2.0 * math.pi)
    if t < 0.0:
        t += 2.0 * math.pi
    t -= math.pi
    if t == -math.pi:
        t = math.pi
    return t


def sinc_unnormalized(z: float) -> float:
    """sinc(z) = 1 if |z| < 1e-12 else sin(z)/z (not numpy.sinc)."""
    if abs(z) < 1e-12:
        return 1.0
    return math.sin(z) / z


def step_unicycle(
    state: Pose,
    v: float,
    omega: float,
    dt: float,
) -> Pose:
    """Integrate constant (v, omega) over dt using the unnormalized sinc form."""
    x, y, theta = state
    if not all(math.isfinite(c) for c in state):
        return state
    if not math.isfinite(v) or not math.isfinite(omega) or not math.isfinite(dt):
        return (float("nan"), float("nan"), float("nan"))
    if dt <= 0.0:
        return (float("nan"), float("nan"), float("nan"))
    dtheta = omega * dt
    s = sinc_unnormalized(dtheta / 2.0)
    x_new = x + v * dt * s * math.cos(theta + dtheta / 2.0)
    y_new = y + v * dt * s * math.sin(theta + dtheta / 2.0)
    theta_new = wrap(theta + dtheta)
    return (x_new, y_new, theta_new)


def apply_command(
    v: float,
    omega: float,
    prev: Command,
    dt: float,
) -> tuple[float, float, StepStatus]:
    """Clip and rate-limit a raw command toward the previous applied command."""
    if not math.isfinite(v) or not math.isfinite(omega) or not math.isfinite(dt):
        return (float("nan"), float("nan"), "invalid")
    if dt <= 0.0:
        return (float("nan"), float("nan"), "invalid")
    v_clip = max(-V_MAX, min(V_MAX, v))
    w_clip = max(-OMEGA_MAX, min(OMEGA_MAX, omega))
    pv, pw = prev
    dv_max = A_MAX * dt
    dw_max = ALPHA_MAX * dt
    v_app = pv + max(-dv_max, min(dv_max, v_clip - pv))
    w_app = pw + max(-dw_max, min(dw_max, w_clip - pw))
    v_app = max(-V_MAX, min(V_MAX, v_app))
    w_app = max(-OMEGA_MAX, min(OMEGA_MAX, w_app))
    return (v_app, w_app, "ok")


def pose_valid(x: float, y: float, theta: float) -> bool:
    """Pose is invalid if non-finite or outside the 10 m world."""
    if not all(math.isfinite(c) for c in (x, y, theta)):
        return False
    from nav_imitation_learning.constants import MAP_SIZE_M

    if x < 0.0 or x > MAP_SIZE_M or y < 0.0 or y > MAP_SIZE_M:
        return False
    return True


def state_array(state: Pose) -> npt.NDArray[np.float64]:
    return np.array(state, dtype=np.float64)
