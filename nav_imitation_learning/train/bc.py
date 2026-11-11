# SPDX-License-Identifier: Apache-2.0
"""Behavior cloning on expert demonstrations."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from nav_imitation_learning.constants import BATCH_SIZE, BC_EPOCHS, TRAIN_SEED
from nav_imitation_learning.policy.mlp import adam_step, mlp_backward, mlp_forward, new_adam_moments


def train_bc(
    params: dict[str, npt.NDArray[np.float64]],
    obs: npt.NDArray[np.float64],
    actions: npt.NDArray[np.float64],
    epochs: int = BC_EPOCHS,
    seed: int = TRAIN_SEED,
) -> dict[str, npt.NDArray[np.float64]]:
    if not np.all(np.isfinite(obs)) or not np.all(np.isfinite(actions)):
        raise ValueError("non-finite training batch")
    n = obs.shape[0]
    moments = new_adam_moments(params)
    t_step = 0
    for epoch in range(epochs):
        rng = np.random.Generator(np.random.PCG64(seed + epoch))
        indices = rng.permutation(n)
        for start in range(0, n, BATCH_SIZE):
            batch_idx = indices[start : start + BATCH_SIZE]
            if batch_idx.size == 0:
                continue
            t_step += 1
            xb = obs[batch_idx]
            yb = actions[batch_idx]
            pre, _ = mlp_forward(params, xb)
            diff = pre - yb
            grad_pre = (2.0 / (xb.shape[0] * 2.0)) * diff
            grads = mlp_backward(params, xb, grad_pre)
            adam_step(params, grads, moments, t_step)
    return params
