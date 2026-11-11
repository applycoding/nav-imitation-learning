# SPDX-License-Identifier: Apache-2.0
"""MLP initialization, forward pass, Adam, and gradient checks."""

from __future__ import annotations

import math

import numpy as np
import pytest

from nav_imitation_learning.constants import OBS_DIM
from nav_imitation_learning.policy.mlp import (
    adam_step,
    glorot_limit,
    mlp_backward,
    mlp_forward,
    mlp_init,
    new_adam_moments,
)


def test_glorot_bounds() -> None:
    limits = {
        "W1": glorot_limit(40, 64),
        "W2": glorot_limit(64, 64),
        "W3": glorot_limit(64, 2),
    }
    params = mlp_init()
    for key, lim in limits.items():
        assert np.all(np.abs(params[key]) <= lim + 1e-15)
        assert np.var(params[key]) > 1e-6
    for bkey in ("b1", "b2", "b3"):
        assert np.all(params[bkey] == 0.0)


def test_forward_shape() -> None:
    params = mlp_init()
    x = np.zeros((5, OBS_DIM), dtype=np.float64)
    pre, cmd = mlp_forward(params, x)
    assert pre.shape == (5, 2)
    assert cmd.shape == (5, 2)


def test_adam_one_step() -> None:
    params = {"w": np.array([0.0], dtype=np.float64)}
    grads = {"w": np.array([1.0], dtype=np.float64)}
    moments = new_adam_moments(params)
    adam_step(params, grads, moments, 1)
    expected = -3e-3 * 1.0 / (math.sqrt(1.0) + 1e-8)
    assert abs(params["w"][0] - expected) < 1e-9


def test_adam_two_step_moment_persist() -> None:
    params = {"w": np.array([0.0], dtype=np.float64)}
    moments = new_adam_moments(params)
    adam_step(params, {"w": np.array([1.0])}, moments, 1)
    w1 = params["w"][0]
    adam_step(params, {"w": np.array([1.0])}, moments, 2)
    assert params["w"][0] != w1


def test_grad_finite_diff() -> None:
    rng = np.random.Generator(np.random.PCG64(0))
    params = mlp_init(rng)
    x = rng.standard_normal((4, OBS_DIM))
    y = rng.standard_normal((4, 2))
    pre, _ = mlp_forward(params, x)
    diff = pre - y
    grad_pre = (2.0 / (x.shape[0] * 2.0)) * diff
    analytic = mlp_backward(params, x, grad_pre)

    eps = 1e-6
    for key in ("W1", "b1"):
        arr = params[key]
        num = np.zeros_like(arr)
        it = np.nditer(arr, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            idx = it.multi_index
            orig = float(arr[idx])
            arr[idx] = orig + eps
            pre_p, _ = mlp_forward(params, x)
            arr[idx] = orig - eps
            pre_m, _ = mlp_forward(params, x)
            arr[idx] = orig
            loss_p = np.mean((pre_p - y) ** 2)
            loss_m = np.mean((pre_m - y) ** 2)
            num[idx] = (loss_p - loss_m) / (2 * eps)
            it.iternext()
        rel = np.linalg.norm(analytic[key] - num) / (np.linalg.norm(num) + 1e-12)
        assert rel < 1e-6


def test_nonfinite_batch_raises() -> None:
    from nav_imitation_learning.train.bc import train_bc

    params = mlp_init()
    obs = np.ones((2, OBS_DIM))
    act = np.ones((2, 2))
    obs[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        train_bc(params, obs, act, epochs=1)
