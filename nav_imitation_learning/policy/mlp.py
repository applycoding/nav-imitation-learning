# SPDX-License-Identifier: Apache-2.0
"""Two-hidden-layer MLP and Adam optimizer in numpy."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt

from nav_imitation_learning.constants import (
    ADAM_B1,
    ADAM_B2,
    ADAM_EPS,
    ADAM_LR,
    MLP_HIDDEN,
    OBS_DIM,
    OMEGA_MAX,
    V_MAX,
    WEIGHT_SEED,
)


def glorot_limit(fan_in: int, fan_out: int) -> float:
    return math.sqrt(6.0 / (fan_in + fan_out))


def mlp_init(rng: np.random.Generator | None = None) -> dict[str, npt.NDArray[np.float64]]:
    if rng is None:
        rng = np.random.Generator(np.random.PCG64(WEIGHT_SEED))
    params: dict[str, npt.NDArray[np.float64]] = {}
    layers = [(OBS_DIM, MLP_HIDDEN), (MLP_HIDDEN, MLP_HIDDEN), (MLP_HIDDEN, 2)]
    names_w = ["W1", "W2", "W3"]
    names_b = ["b1", "b2", "b3"]
    for (fan_in, fan_out), nw, nb in zip(layers, names_w, names_b, strict=True):
        a = glorot_limit(fan_in, fan_out)
        params[nw] = rng.uniform(-a, a, size=(fan_in, fan_out)).astype(np.float64)
        params[nb] = np.zeros(fan_out, dtype=np.float64)
    return params


def mlp_forward(
    params: dict[str, npt.NDArray[np.float64]],
    obs_batch: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    x = obs_batch
    h1 = np.tanh(x @ params["W1"] + params["b1"])
    h2 = np.tanh(h1 @ params["W2"] + params["b2"])
    pre_clip = h2 @ params["W3"] + params["b3"]
    clipped = np.clip(pre_clip, -1.0, 1.0)
    cmd = np.empty_like(clipped)
    cmd[:, 0] = clipped[:, 0] * V_MAX
    cmd[:, 1] = clipped[:, 1] * OMEGA_MAX
    return pre_clip, cmd


def mlp_backward(
    params: dict[str, npt.NDArray[np.float64]],
    obs_batch: npt.NDArray[np.float64],
    grad_pre_clip: npt.NDArray[np.float64],
) -> dict[str, npt.NDArray[np.float64]]:
    x = obs_batch
    h1 = np.tanh(x @ params["W1"] + params["b1"])
    h2 = np.tanh(h1 @ params["W2"] + params["b2"])
    dh2 = grad_pre_clip @ params["W3"].T
    dh2 *= 1.0 - h2 * h2
    dh1 = dh2 @ params["W2"].T
    dh1 *= 1.0 - h1 * h1
    grads = {
        "W3": h2.T @ grad_pre_clip,
        "b3": grad_pre_clip.sum(axis=0),
        "W2": h1.T @ dh2,
        "b2": dh2.sum(axis=0),
        "W1": x.T @ dh1,
        "b1": dh1.sum(axis=0),
    }
    return grads


def new_adam_moments(params: dict[str, npt.NDArray[np.float64]]) -> dict[str, Any]:
    m: dict[str, npt.NDArray[np.float64]] = {}
    v: dict[str, npt.NDArray[np.float64]] = {}
    for key, arr in params.items():
        m[key] = np.zeros_like(arr)
        v[key] = np.zeros_like(arr)
    return {"m": m, "v": v, "t": 0}


def adam_step(
    params: dict[str, npt.NDArray[np.float64]],
    grads: dict[str, npt.NDArray[np.float64]],
    moments: dict[str, Any],
    t: int,
) -> None:
    m = moments["m"]
    v = moments["v"]
    for key in params:
        g = grads[key]
        m[key] = ADAM_B1 * m[key] + (1.0 - ADAM_B1) * g
        v[key] = ADAM_B2 * v[key] + (1.0 - ADAM_B2) * (g * g)
        mhat = m[key] / (1.0 - ADAM_B1**t)
        vhat = v[key] / (1.0 - ADAM_B2**t)
        params[key] = params[key] - ADAM_LR * mhat / (np.sqrt(vhat) + ADAM_EPS)


def save_params(path: str, params: dict[str, npt.NDArray[np.float64]], extra: dict[str, Any] | None = None) -> None:
    payload = {k: v for k, v in params.items()}
    if extra:
        for k, val in extra.items():
            if isinstance(val, (int, float, str)):
                payload[k] = np.array(val)
            else:
                payload[k] = val
    np.savez(path, allow_pickle=False, **payload)


def load_params(path: str) -> dict[str, npt.NDArray[np.float64]]:
    data = np.load(path)
    keys = ["W1", "b1", "W2", "b2", "W3", "b3"]
    return {k: np.asarray(data[k], dtype=np.float64) for k in keys}
