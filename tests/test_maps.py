# SPDX-License-Identifier: Apache-2.0
"""Fixture integrity and map generator regression tests."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np

from nav_imitation_learning.constants import ID_TEST_SEEDS, OOD_TEST_SEEDS, TRAIN_SEEDS
from nav_imitation_learning.env.maps import (
    family_for_seed,
    generate_map,
    occupancy_sha256,
    world_from_fixture,
    write_fixture,
)

ALL_SEEDS = list(TRAIN_SEEDS) + list(ID_TEST_SEEDS) + list(OOD_TEST_SEEDS)


def _seed_dir(fixtures_root: Path, seed: int) -> Path:
    family = family_for_seed(seed)
    return fixtures_root / family / str(seed)


def test_fixture_hashes(fixtures_root: Path) -> None:
    for seed in ALL_SEEDS:
        d = _seed_dir(fixtures_root, seed)
        meta = json.loads((d / "meta.json").read_text())
        occ = np.load(d / "occupancy.npy")
        sidecar = (d / "occupancy.sha256").read_text().strip()
        assert occ.shape == (200, 200)
        assert occ.dtype == np.uint8
        assert occupancy_sha256(occ) == sidecar
        assert meta["occupancy_sha256"] == sidecar
        assert meta["boxes"] == meta["boxes"]


def test_fixture_separation_and_shape(fixtures_root: Path) -> None:
    for seed in ALL_SEEDS:
        w = world_from_fixture(_seed_dir(fixtures_root, seed))
        dist = math.hypot(w.goal[0] - w.start[0], w.goal[1] - w.start[1])
        assert 3.0 <= dist <= 8.0
        assert w.occupancy.shape == (200, 200)


def test_regenerate_matches_committed(fixtures_root: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp)
        for seed in ALL_SEEDS:
            family = family_for_seed(seed)
            committed = world_from_fixture(fixtures_root / family / str(seed))
            regen = generate_map(seed, family)
            assert regen is not None
            assert regen.occupancy_sha256 == committed.occupancy_sha256
            assert regen.boxes == committed.boxes
            assert list(regen.start) == list(committed.start)
            assert list(regen.goal) == list(committed.goal)


def test_write_roundtrip(tmp_path: Path) -> None:
    w = world_from_fixture(
        Path(__file__).resolve().parent.parent / "fixtures" / "id_test" / "21001"
    )
    write_fixture(w, tmp_path / "21001")
    w2 = world_from_fixture(tmp_path / "21001")
    assert w2.occupancy_sha256 == w.occupancy_sha256
