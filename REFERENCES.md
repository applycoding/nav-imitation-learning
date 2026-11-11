# References

Permissively licensed GitHub projects used as **references only**. Do not copy sources, tests, worlds, training loops, or documentation prose. Credit any conceptually derived idea in the README. Licenses were checked on 2026-11-11 with `gh api repos/OWNER/REPO --jq .license.spdx_id` and with the raw LICENSE file when the SPDX API was not enough.

**Outbound license for this demo repo is Apache-2.0**, not MIT. A LICENSE file with the official Apache-2.0 text, a NOTICE file crediting Arpan Bom, and SPDX-License-Identifier: Apache-2.0 headers are required at build time.

This demo must not duplicate existing applycoding repos. The closest is [applycoding/robot-learning-vla-demo](https://github.com/applycoding/robot-learning-vla-demo) (SimReach: language-conditioned 2-link arm, BC vs PPO vs VLA, CPU only; outbound **MIT** per `gh api` SPDX). This repo is planar navigation, lidar-like observations, BC vs DAgger, numpy only, Apache-2.0. Do not copy that tree.

## Selected references (4)

### 1. Farama-Foundation/Gymnasium

- URL: https://github.com/Farama-Foundation/Gymnasium
- License: **MIT** (`gh api` SPDX: MIT; LICENSE: https://github.com/Farama-Foundation/Gymnasium/blob/main/LICENSE)
- Borrow: the *idea* of a small reset/step environment with observations, actions, and a termination flag. v1 implements a custom env and does **not** depend on or vendor Gymnasium.

### 2. HumanCompatibleAI/imitation

- URL: https://github.com/HumanCompatibleAI/imitation
- License: **MIT** (`gh api` SPDX: MIT; LICENSE: https://github.com/HumanCompatibleAI/imitation/blob/master/LICENSE)
- Borrow: the *idea* of behavior cloning and a dataset-aggregation loop. Do not vendor, wrap, or translate this library.

### 3. AtsushiSakai/PythonRobotics

- URL: https://github.com/AtsushiSakai/PythonRobotics
- License: **MIT**. The repository LICENSE file is the MIT License (https://github.com/AtsushiSakai/PythonRobotics/blob/master/LICENSE, HTTP 200 on 2026-11-11, text begins "The MIT License (MIT)"). GitHub's license API reports `NOASSERTION` / `Other`; that API value is incomplete, not a different license. Treat the project as MIT and do not copy sources.
- Borrow: grid-planning *ideas* only. The A* in this demo is a new implementation with the tie-breaks and costs locked in PLAN.md.

### 4. huggingface/lerobot

- URL: https://github.com/huggingface/lerobot
- License: **Apache-2.0** (`gh api` SPDX: Apache-2.0; LICENSE: https://github.com/huggingface/lerobot/blob/main/LICENSE)
- Borrow: the *idea* of a small, documented robot-learning stack with recorded evaluations. This demo is deliberately smaller, navigation-only, and numpy-only. Do not vendor LeRobot, its datasets, or its models.

## Academic citations (not code)

- Ross, Gordon, and Bagnell, "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning," AISTATS 2011. https://proceedings.mlr.press/v15/ross11a.html and https://proceedings.mlr.press/v15/ross11a/ross11a.pdf (both HTTP 200 on 2026-11-11).
- Hart, Nilsson, and Raphael, "A Formal Basis for the Heuristic Determination of Minimum Cost Paths," IEEE TSSC, 1968. A* citation only.
- Amanatides and Woo, "A Fast Voxel Traversal Algorithm for Ray Tracing," Eurographics 1987. Algorithm citation only. A commonly linked PDF host was not used as a pin after a TLS failure on 2026-11-11.
- Coulter, "Implementation of the Pure Pursuit Path Tracking Algorithm," CMU-RI-TR-92-01, 1992. Related *idea* only. This demo implements a lookahead heading controller, not Coulter's curvature law. https://www.ri.cmu.edu/pub_files/pub3/coulter_r_craig_1992_1/coulter_r_craig_1992_1.pdf redirected with HTTP 200 on 2026-11-11 to the Robotics Institute publications host.

## Idea-only or exclude

- ros-navigation/navigation2 - mixed licenses; not a dependency. Do not vendor or copy it.
- Stable-Baselines3, PyTorch, TensorFlow, JAX - not used in v1.
- applycoding/robot-learning-vla-demo - MIT, different domain. Do not copy.
- applycoding/nav2-coverage-planner-compare - do not import or copy the `ncp` planner library.
- GPL or LGPL planning or learning libraries - not dependencies.

Direct runtime and dev dependencies (numpy, pytest, ruff, mypy, Vite, TypeScript, Playwright, and the rest of the lockfiles) belong in NOTICE with version and license once installed. They are dependencies, not idea references.
