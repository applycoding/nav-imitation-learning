# SPDX-License-Identifier: Apache-2.0
"""Label-shuffle and zero-policy mutants."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from nav_imitation_learning.constants import SHUFFLE_MUTANT_SEED
from nav_imitation_learning.env.plant import Command, Pose
from nav_imitation_learning.eval.closed_loop import PolicyFn
from nav_imitation_learning.train.bc import train_bc as train_bc_fn


def shuffle_mutant_train(
    init_params: dict[str, npt.NDArray[np.float64]],
    obs: npt.NDArray[np.float64],
    actions: npt.NDArray[np.float64],
) -> dict[str, npt.NDArray[np.float64]]:
    rng = np.random.Generator(np.random.PCG64(SHUFFLE_MUTANT_SEED))
    perm = rng.permutation(obs.shape[0])
    shuffled = actions[perm].copy()
    return train_bc_fn({k: v.copy() for k, v in init_params.items()}, obs, shuffled)


def zero_policy_fn() -> PolicyFn:
    def _zero(
        obs: npt.NDArray[np.float32],
        pose: Pose,
        prev: Command,
        goal: tuple[float, float],
    ) -> tuple[float, float]:
        del obs, pose, prev, goal
        return 0.0, 0.0  # applied command; swept_step applies zero motion

    return _zero
