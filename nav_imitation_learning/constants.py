# SPDX-License-Identifier: Apache-2.0
"""Locked simulation constants from PLAN.md."""

from __future__ import annotations

import math

ROBOT_RADIUS_M: float = 0.20
PLAN_INFLATION_M: float = 0.24
V_MAX: float = 0.60
OMEGA_MAX: float = 1.20
A_MAX: float = 1.00
ALPHA_MAX: float = 3.00
DT: float = 0.05
GOAL_TOL_XY: float = 0.25
T_MAX: float = 40.0
T_MAX_STEPS: int = 800
MAP_SIZE_M: float = 10.0
MAP_RES_M: float = 0.05
MAP_CELLS: int = 200
BORDER_CELLS: int = 4
BORDER_M: float = BORDER_CELLS * MAP_RES_M
R_MAX: float = 4.0
N_RAYS: int = 36
OBS_DIM: int = 40
LOOKAHEAD_M: float = 0.50
HEADING_KP: float = 1.5
V_CRUISE: float = 0.50
WEIGHT_SEED: int = 20261111
TRAIN_SEED: int = 20261111
MLP_HIDDEN: int = 64
BC_EPOCHS: int = 40
DAGGER_EPOCHS: int = 20
DAGGER_ROUNDS: int = 4
BATCH_SIZE: int = 64
ADAM_LR: float = 3e-3
ADAM_B1: float = 0.9
ADAM_B2: float = 0.999
ADAM_EPS: float = 1e-8
SHUFFLE_MUTANT_SEED: int = 20261111

TRAIN_SEEDS: tuple[int, ...] = tuple(range(11001, 11021))
ID_TEST_SEEDS: tuple[int, ...] = tuple(range(21001, 21009))
OOD_TEST_SEEDS: tuple[int, ...] = tuple(range(31001, 31009))
DEV_SEEDS: tuple[int, ...] = tuple(range(41001, 41009))
DEV_OOD_SEEDS: tuple[int, ...] = tuple(range(42001, 42009))

SQRT2: float = math.sqrt(2.0)
PI: float = math.pi
