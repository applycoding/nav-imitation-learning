# SPDX-License-Identifier: Apache-2.0
"""Expert success oracle on committed train, id_test, and ood_test maps."""

from __future__ import annotations

from nav_imitation_learning.expert.rollout import expert_episode


def _assert_suite(worlds: list, expected_n: int) -> None:
    successes = collisions = invalids = 0
    for w in worlds:
        ep = expert_episode(w, w.start, w.goal)
        if ep["status"] == "success":
            successes += 1
        if ep.get("collision"):
            collisions += 1
        if ep["status"] == "invalid":
            invalids += 1
    assert len(worlds) == expected_n
    assert successes == expected_n
    assert collisions == 0
    assert invalids == 0


def test_expert_train_oracle(train_worlds) -> None:
    _assert_suite(train_worlds, 20)


def test_expert_id_oracle(id_test_worlds) -> None:
    _assert_suite(id_test_worlds, 8)


def test_expert_ood_oracle(ood_test_worlds) -> None:
    _assert_suite(ood_test_worlds, 8)
