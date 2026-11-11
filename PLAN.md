# nav-imitation-learning - v1 plan

Revised 2026-11-11 after independent review, independent version, URL, license, geometry, and math checks, and a recorded numpy prototype on development maps (see Decision I). Outbound license is **Apache-2.0**. Planning only; this document does not create a public repo or implementation.

`gh repo list applycoding --limit 200` on 2026-11-11 shows no `applycoding/nav-imitation-learning`.

## Goal

Public portfolio demo for Arpan Bom (idea #9): a CPU-only **simulation** lab for imitation learning in planar navigation.

A scripted expert (grid planner plus a lookahead heading tracker) drives a kinematic differential-drive robot on seeded occupancy maps and records demonstrations. A behavior-cloning (BC) policy maps **local** observations to `(v, omega)`. Closed-loop evaluation on held-out maps is the regression check for covariate shift. Four Dataset Aggregation (DAgger-style) rounds query that same expert at policy-visited states and retrain from the saved Xavier weights. The shipped policy is the train-selected checkpoint among behavior cloning and those four rounds (rule in Decision G). A recorded prototype on development maps backs the locked margins in Decision I. Mutants that shuffle labels, or that command zeros, must fail their gates while still producing healthy finite episodes. A GitHub Pages dashboard at base `/nav-imitation-learning/` renders **recorded CI output**. The page does not train and does not simulate.

This is an original numpy teaching implementation on synthetic maps, inspired by published imitation-learning and classical planning ideas. It is not a production driving policy, not a foundation vision-language-action model, not on-robot data collection, and not certified autonomous navigation. Do not claim a no-regret bound or “the official DAgger code.”

Git author and committer: **Arpan Bom <applycoding@gmail.com>**. No co-author trailers. No credit to any assistant or tool in the README, NOTICE, or commits.

## Scope

In v1:

- Synthetic 10 m × 10 m occupancy maps, locked seed families, exact generator below.
- Kinematic unicycle, 8-connected A* expert, lookahead heading tracker, 40-D local features.
- Numpy MLP, MSE behavior cloning, four aggregation rounds.
- Closed-loop suites, open-loop action MSE on frozen expert states, two mutants.
- Schema-v1 JSON report, Vite + Canvas dashboard, Playwright smoke, one CI workflow.

Do not depend on, submodule, vendor, or copy these. Reimplement the environment, expert, and learner from first principles.

- **applycoding/robot-learning-vla-demo** (MIT outbound; this repo is Apache-2.0): language-conditioned **2-link arm reaching**; HDF5 demos; BC vs PPO vs VLA (tiny encoder and frozen CLIP). This repo is planar differential-drive navigation, lidar-like rays, BC vs DAgger, with no arm, language, PPO, VLA, CLIP, or PyTorch. Inspect that repository at a recorded commit before implementation and do not copy sources or prose.
- applycoding/nav2-coverage-planner-compare - multi-planner coverage bake-off. Here one reimplemented A* is only the scripted expert. Do not import `ncp`.
- applycoding/teb-mpc-local-planner - TEB-style vs OSQP MPC. This repo is imitation learning, not an optimizer bake-off.
- applycoding/nav2-behavior-trees - mission trees.
- applycoding/gazebo-robot-ci, ros2-nav2-slam-sim, ros2-voice-command-robot - Gazebo or Nav2. This demo does not start Gazebo or Nav2.
- applycoding/teleop-console-demo, flutter-robot-operator, compose-android-robot-companion - operator UIs.
- applycoding/multi-robot-warehouse-planner - CBS / MAPF.
- applycoding/ekf-gnss-localization, sim-gimbal-stabilizer, rosbag-eval-bench, field-telemetry-analyzer, lidar-shape-classifier-ros2 - other domains.

Gymnasium, HumanCompatibleAI/imitation, PythonRobotics, LeRobot, Nav2, and Stable-Baselines3 are citations only. Do not vendor or translate their sources.

## Architecture

### Decision A - numpy MLP, no PyTorch, no ROS in v1

One installable package `nav_imitation_learning`, console script `nil`, build backend **hatchling 1.33.0**. Training, evaluation, and tests use NumPy only: a two-hidden-layer MLP and Adam written out in numpy. Core CI installs no ROS, Gazebo, PyTorch, TensorFlow, or JAX, and downloads no network weights.

Why: CI stays CPU-light and license-simple, and the demo does not retread SimReach's PyTorch / PPO / VLA stack. Optional ROS 2 is out of scope for v1.

### Decision B - kinematic unicycle, two geometries

Frame: planar `map`, +x East, +y North, yaw about +z (REP-103-like ENU). State `(x, y, theta)` in meters and radians. No physics engine. World is `[0, 10] × [0, 10]`. A pose with non-finite coordinates, or with `x` or `y` outside `[0, 10]`, is `invalid`.

Locked constants:

| Symbol | Value |
| --- | --- |
| `robot_radius_m` (physical disk, contact) | 0.20 |
| `plan_inflation_m` (planning mask only) | 0.24 |
| `v_max` | 0.60 m/s |
| `omega_max` | 1.20 rad/s |
| `a_max` | 1.00 m/s² |
| `alpha_max` | 3.00 rad/s² |
| `dt` | 0.05 s (plant and control share this step) |
| `goal_tol_xy` | 0.25 m |
| `T_max` | 40 s = 800 steps |
| Heading at the goal | not required |

Integration of a constant `(v, omega)` over `dt` uses the unnormalized sinc form, `sinc(z) = 1` if `|z| < 1e-12` else `sin(z)/z`. This is **not** `numpy.sinc`, which is `sin(πz)/(πz)`. The equivalent numpy call, if used, is `np.sinc(dθ / (2π))`. Let `dθ = ω dt` and `s = sinc(dθ / 2)`:

```
x' = x + v * dt * s * cos(θ + dθ / 2)
y' = y + v * dt * s * sin(θ + dθ / 2)
θ' = wrap(θ + dθ)   # to (-π, π]
```

Algebra matches the classical arc map: `Δx = (v/ω) (sin(θ+dθ) - sin θ)` when `ω ≠ 0`. Wrap: reduce modulo `2π` into `[-π, π)`, then map `-π` to `+π`, so `wrap(π) = π` and `wrap(-π) = π`. Reject non-finite angles before wrapping. `dt <= 0` or non-finite `dt` is `invalid` and does not write a successor.

Command pipeline, **once per real transition**: reject non-finite `(v, omega)` as `invalid`; clip to `|v| ≤ v_max` and `|ω| ≤ omega_max`; then rate-limit toward the previous **applied** command with `|Δv| ≤ a_max dt = 0.05` and `|Δω| ≤ alpha_max dt = 0.15`. Previous command starts at `(0, 0)`. The core never reads a wall clock. The only RNG is `numpy.random.Generator(numpy.random.PCG64(seed))`.

**Contact is continuous and independent of the planning mask.** Clearance is the Euclidean distance from the disk center to the union of the **original** occupied rectangles (including the border wall) minus `robot_radius_m`. Distance to an axis-aligned rectangle is the standard point-AABB distance (zero when the point is inside). `clearance < 0` is a collision. `clearance == 0` is not. Compare in float64 against `0.0`, not against a cell mask. A start pose must have clearance `>= 0` and must not be `invalid`.

**Swept contact.** Endpoints alone are not enough. After computing the exact successor, also test clearance at `K` intermediate arc samples, `K = max(2, ceil(|v| * dt / 0.01))`, at equal fractions of `dt` (the same sinc map on `k/K * dt`). If any sample or either endpoint has `clearance < 0`, the episode status is `collision`. The stored successor is the last in-world sample **before** penetration if one exists; if the start of the step is already penetrating, status is `collision` and the pose is left unchanged. Collision takes precedence over goal and timeout on the same step.

**Planning mask.** A cell is plan-lethal when the distance from its **center** to the union of original rectangles is `< plan_inflation_m`. This mask is for A* only. It is deliberately conservative (`0.24 > 0.20`) so a cell-center path has a margin against continuous contact and against tracker corner-cutting. Do not claim the mask equals continuous contact.

### Decision C - synthetic maps, exact generator

World: 10 m × 10 m, resolution 0.05 m, **200 × 200** `uint8` (1 = occupied, C order). Cell `(i, j)` covers `[i*res, (i+1)*res) × [j*res, (j+1)*res)`, origin at the south-west corner. The border wall is **4 cells** (0.20 m) on every side. Snap every generated coordinate to the grid: `snap(v) = floor(v / res + 1e-12) * res`, then snap width and height so the painted cells exactly match the rectangle. Same seed produces a bit-identical occupancy array: sha256 of `occupancy.tobytes()` in C order. Maps are not floorplans, SLAM exports, or customer sites.

AABB edge gap between rectangles `A` and `B`:

```
dx = max(0, A.x - (B.x+B.w), B.x - (A.x+A.w))
dy = max(0, A.y - (B.y+B.h), B.y - (A.y+A.h))
if A and B overlap on both axes: gap = 0
elif they overlap on one axis: gap = the other axis gap
else: gap = hypot(dx, dy)
```

The inner face of the border is the square `[0.20, 9.80]²`. Gap from a rectangle to the border is the minimum of its distances to those four inner edges (0 if it crosses an inner face).

**Base-map draw**, `Generator(PCG64(seed))` only, in this order:

1. Zero the grid and paint the border.
2. `n_boxes = int(rng.integers(5, 9))`  # 5, 6, 7, or 8
3. For `k` in `range(n_boxes)`: up to 80 attempts. Each attempt **always** draws, in order: `w = 0.50 + rng.random() * 2.00`, `h = 0.50 + rng.random() * 2.00`, `x = 0.20 + rng.random() * max(0, 9.80 - w - 0.20)`, `y = 0.20 + rng.random() * max(0, 9.80 - h - 0.20)`, then snap `x,y,w,h`. If `w < 0.50` or `h < 0.50` after snap, reject. If gap to every accepted box and to the border inner face is `>= min_gap`, accept, paint, and stop attempting this `k`. Otherwise reject and continue. If 80 attempts fail, skip this box and record `base_box_skips += 1`.
4. `min_gap` is **1.20 m** on `train`, `id_test`, and `dev`; **0.90 m** on `ood` base maps (extra boxes come later).

**Start and goal**, same generator, after the base paint:

- Build the planning mask. Collect cell centers of all non-plan-lethal cells whose continuous clearance (physical radius) is `>= 0.0`. If fewer than 16 such cells, the generator attempt for this seed fails.
- For up to 50 pair attempts, always draw: two independent `rng.integers(0, n_free)` indices, then `yaw = wrap(float(rng.uniform(-π, π)))`. Start is the first center plus that yaw; goal is the second center. Accept iff Euclidean separation is in `[3.0, 8.0]` m, A* on the planning mask returns a path, **and** a full expert episode from that pose succeeds (continuous contact, not only A* connectivity). Otherwise reject.
- If 50 attempts fail, the generator fails the fixture test for that seed (do not silently ship an unsolvable map).

**Families:**

| Family | Seeds | Role |
| --- | --- | --- |
| train | `11001 .. 11020` | Demonstrations and DAgger rollouts |
| id_test | `21001 .. 21008` | Held-out, same generator, never used to choose the generator or hyperparameters |
| ood_test | `31001 .. 31008` | Held-out OOD, never used to choose the generator or hyperparameters |
| dev | `41001 .. 41008` plus optional `dev_ood` `42001 .. 42008` | Prototype and generator/hyperparameter work only. Not a ship gate. Not shown on Pages |

**OOD extra boxes** (ood_test and dev_ood only), after a start/goal pair is accepted on the base map. Compute the expert polyline for that pair. For two boxes of snapped size **0.60 × 0.60** m:

- For each box, up to 50 attempts. Each attempt draws `t = rng.random()` and an offset `s = (rng.random() - 0.5) * 0.80`. Place the box center at the point a fraction `t` along the polyline (arc length), then offset perpendicular to the local tangent by `s`, snap, and clip so the box stays inside the inner face.
- Temporarily paint. Accept iff the box occupies **at least one** A* cell of the original expert path, A* still connects start to goal on the new planning mask, and a full expert episode still succeeds. Then keep the paint.
- Otherwise revert and retry. If 50 attempts fail, skip and record `extra_box_skips += 1`.

Two skips are allowed. Expert success on the shipped fixture is still required. The fixture record is: `seed`, `family`, `start_xytheta`, `goal_xy`, `occupancy_sha256`, `boxes` as snapped `[x,y,w,h]`, `base_box_skips`, `extra_box_skips`.

**No test-set tuning.** If a prototype on **dev** maps does not show the intended BC/DAgger narrative, change hyperparameters or the generator, then freeze, and evaluate on the already-locked id/ood seed lists. If those locked lists still miss the narrative, report the honest numbers. Do **not** edit id_test or ood_test maps, and do not replace those seeds, to chase a gap. A generator change after the first locked evaluation requires a **new unused** seed list, recorded in a plan amendment, and a statement that the previous list was retired.

`nil generate` writes fixtures; they are committed. CI regenerates into a temp directory and requires identical occupancy sha256, box lists, and start/goal.

### Decision D - scripted expert: A* plus a lookahead heading tracker

Planner: 8-connected A* on the **planning mask**. Step cost `1` orthogonal and `sqrt(2)` diagonal, in cell units. Heuristic is octile `h = max(dx, dy) + (sqrt(2) - 1) * min(dx, dy)`. A diagonal is illegal when either orthogonal neighbor is plan-lethal. Tie-break: lowest `f`, then lowest `g`, then lowest insertion counter. Neighbors expand in order E, NE, N, NW, W, SW, S, SE. The polyline is cell centers, with the first point replaced by the continuous start `(x, y)` and the last by the continuous goal, provided each replacement segment stays in non-plan-lethal space; otherwise keep the cell center and let the tracker finish.

Tracker: this is a **lookahead heading controller**, inspired by the pure-pursuit *idea* of tracking a point ahead on the path. It is **not** Coulter's curvature law `κ = 2 sin(e) / L`. Do not call the implemented law “pure pursuit” in the README.

Lookahead **0.50 m** along the polyline from the robot's projection onto the remaining path. Persistent progress: the projection may not move backward along path arc length (tie-break: smallest arc length among nearest points). `v_cruise = 0.50`. Heading error `e` to the lookahead point, wrapped to `(-π, π]`. `heading_kp = 1.5`.

```
if hypot(goal - xy) <= goal_tol_xy:  raw = (0, 0)
elif |e| > π/2:                      v_raw = 0;  ω_raw = heading_kp * e
elif dist_to_goal < 0.50:            v_raw = v_cruise * dist_to_goal / 0.50; ω_raw = heading_kp * e
else:                                 v_raw = v_cruise; ω_raw = heading_kp * e
```

Then `apply_command` **once**. **Expert safety overlay:** if the swept successor under that applied command would collide, replace the applied command with `(0, clip(heading_kp * e))` after rate limits and test again; if that still collides, apply `(0, 0)` after rate limits. If even a brake still begins in contact, the expert episode is a collision (and the generator must not accept that pair).

The expert is deterministic given map, start, goal, and the previous applied command. A unit-tested expert reaches the goal with zero collisions and zero `invalid` on **every** train, id_test, and ood_test fixture. That 100% is an oracle.

Teacher labels (demonstrations and DAgger) are the **applied** commands actually sent, produced by a **stateless query** `label(world, pose, prev_applied, goal) -> ('ok', v, omega) | ('unlabelable', reason)`. The query must not mutate plant or progress state of a live episode. If A* returns `None`, or the pose is already in contact or invalid, the result is `unlabelable`. Never replace that with a zero action silently.

### Decision E - local observations, dimension 40

The policy does not receive the occupancy grid, the A* polyline, or the map id. Feature vector, `float32`, locked order:

1. **36 range rays.** Beam angles `k * (2π/36)` for `k = -18 .. 17` (from `-π` to `π - 2π/36`). World angle `φ = θ + beam_angle`. Range is the distance to the first occupied **original** cell (before inflation) or to the map border, clipped to `[0, R_max]`, `R_max = 4.0` m, then divided by `R_max`.
2. **Robot-frame goal.** `gx = cosθ (Gx-x) + sinθ (Gy-y)`, `gy = -sinθ (Gx-x) + cosθ (Gy-y)`, each clipped to `[-R_max, R_max]` and divided by `R_max`.
3. **Last applied command.** `v / v_max`, `ω / omega_max`.

Raycast is Amanatides-Woo (1987) on the original occupancy grid. Direction is a unit vector. `t` is in meters. If the start cell is occupied, range is 0. If a direction component has absolute value `< 1e-15`, that axis is parallel (`tDelta = +inf`). The hit range is the entry `t` into the first occupied cell (cell **edge**, not center). If the ray leaves the grid first, the range is the `t` of the boundary crossing. Feature value is `min(range, R_max) / R_max`. An all-ones scan is valid empty space within `R_max`.

**Grid contract.** Cells are half-open `[i res, (i+1) res)`. A point on the east or north face belongs to the next cell if that cell exists, otherwise it is a grid exit. When `tMaxX` and `tMaxY` compare equal, step **X first**, then Y on the following iteration (a documented convention; tested on a diagonal corner). Zero-width tangency along a face of an occupied cell **does** count as a hit (the entry `t` equals the face). Implement and test all four quadrants, both axis-parallel signs, a start on a grid line travelling negative, grid exit, `R_max` equality, the border wall, NaN origin, and out-of-bounds origin (`invalid` observation → episode `invalid`).

### Decision F - MLP `40 → 64 → 64 → 2`

Hidden activations `tanh`. Output layer linear, then clip each component to `[-1, 1]` and scale by `(v_max, omega_max)` to produce the raw policy command (then `apply_command` once in the plant). No softmax and no discrete actions. Training weights and CI artifacts are **float64**. A float32 export is optional and is not used for ship gates.

Initialization, independent `Generator(PCG64(weight_seed))` with `weight_seed = 20261111`: Glorot uniform `a = sqrt(6 / (fan_in + fan_out))`, `W ~ U(-a, a)`, `b = 0`, weights then biases, layers in forward order, via `Generator.uniform`.

Loss, locked: `L = mean((pre_clip - target)**2)` over the full `[N, 2]` tensor (average over batch and both action components). `target` is expert applied `(v/v_max, ω/omega_max)`. Gradients are the analytic backprop of that mean; a test checks them by central finite differences on a two-layer nonzero batch (`rel error < 1e-6`). Optimizer: Adam in numpy, `lr = 3e-3`, `β1 = 0.9`, `β2 = 0.999`, `ε = 1e-8` **outside** the square root, bias correction, `t` increment **once per minibatch** starting at 1. Batch **64**; a remainder smaller than 64 is kept. Each epoch shuffles **row indices** with `Generator(PCG64(train_seed + epoch))`, `train_seed = 20261111`, and walks those indices. Initial BC fit: **40** epochs. No autograd library. One vectorized forward pass serves training and rollout.

### Decision G - DAgger-style loop is in v1

Inspired by Ross, Gordon, and Bagnell, AISTATS 2011. Teaching implementation, not their official code, no no-regret claim.

- Round 0: BC for 40 epochs on expert demonstrations from all 20 train maps (stored start and goal). Save the Xavier initial weights and the 40-epoch BC weights.
- Rounds 1..4: roll out the **current** policy closed-loop on those 20 train episodes (`β = 0`; no expert action is executed). At each **pre-transition** state, query `label(...)`. If `ok`, append `(obs, expert_action_normalized)`. If `unlabelable`, skip the row and increment `unlabelable_count`. Stop the episode on collision, goal, or `T_max` as usual.
- Retrain **from the saved round-0 Xavier weights**, Adam moments reset, **20** epochs, same hyperparameters, on the aggregated dataset (expert rows first, then round-1 rows, then round-2, and so on). Uniform row weighting.
- The policy rolled out in the next aggregation round is that just-trained 20-epoch aggregate fit.
- After behavior cloning and after each of the four rounds, evaluate closed-loop success on the **20 train maps** (never on dev, id_test, or ood_test). Save each checkpoint. The **shipped** policy (the DAgger artifact and the policy used in the ship report) is the candidate with the highest train success count; ties break by fewer train collisions, then by the later round index (behavior cloning is round 0). Record `dagger_selected_round` (0..4) in the report. This selection rule exists because a prototype on dev maps showed that always shipping round 4 can lose a real mid-loop OOD gain.

A development-only equal-budget BC baseline (60 epochs on expert data only, never shipped as a gate) may be logged in the prototype note to separate “more epochs” from “aggregated states.”

### Decision H - episode loop and evaluation

Per-step order, locked:

1. If the current pose is already invalid, terminate `invalid` with no motion.
2. If already in contact, terminate `collision` with no motion.
3. If `hypot(goal - xy) <= goal_tol_xy`, terminate `success` with **no** further motion (`steps` is the number of plant transitions already taken, which is 0 if this holds at reset).
4. If `steps` has already reached 800 without success, terminate `timeout`.
5. Observe, choose a raw command, `apply_command` once, swept-step the plant, then classify: `invalid`, else `collision`, else if at goal `success`, else continue.

`success` requires at-goal, never collided, never `invalid`, and `steps <= 800`. A success on the 800th transition counts. Collision inside the goal disk is a collision, not success.

Episode metrics: `status`, `collision` (bool), `steps`, `time_s = steps * dt`, `path_length_m` (sum of xy chords of applied steps), `final_err_m`, `final_pose`, `start`, `goal`, `seed`. Aggregates, computed from those records, never trusted from a pre-written `pass` flag: `n`, `successes`, `collisions`, `invalids`, `timeouts`, `success_rate = successes / n`, `collision_rate = collisions / n`, `mean_time_success_s` over successes only or JSON `null`.

Suites in the ship report: `expert_id`, `expert_ood`, `bc_id`, `bc_ood`, `dagger_id`, `dagger_ood`, `mutant_shuffle`, `mutant_zero`. Expert-on-train is a CI oracle (20/20) and is not a Pages row. Open-loop MSE of BC, DAgger, and shuffle is computed on frozen expert **id_test** observation rows, on **pre-clip** normalized actions. That MSE is a regression check with a recorded prototype tolerance, not a substitute for closed-loop success.

### Decision I - oracles that cannot be weakened

Integer gates, `n = 8` on test suites. Recompute every gate from trusted episode records in CI; ignore any serialized `pass` field as authority.

**Prototype record, 2026-11-11, seed 20261111.** A numpy-only prototype of the map generator, A* plus lookahead expert, 40-D observations, `40 -> 64 -> 64 -> 2` MLP, 40-epoch BC, four aggregation rounds, and both mutants was run on **dev** maps. Families used: train `11001..11020` (20/20 generated), dev `41001..41008` (8/8), dev_ood `42001..42008` (8/8). The locked `id_test` and `ood_test` lists were not used and were not edited.

Expert: `20/20`, `8/8`, `8/8`, zero collisions, zero `invalid`. Demonstration rows: 4454. Wall time of generate + train + four rounds + evals: **46.1 s** on a local CPU (Python 3.13, numpy 2.5.4), well under the 25-minute `core` budget (CI still adds lint, coverage, and web).

Closed-loop integer counts:

| Policy | train | dev | dev_ood |
| --- | --- | --- | --- |
| BC, 40 epochs | 0/20 (15 collisions) | 2/8 (6 collisions, 0 timeouts, 0 invalid) | 0/8 (8 collisions) |
| Equal-budget BC, 120 epochs, expert rows only | (not the ship gate) | 2/8 | 0/8 |
| DAgger round 1 | 9/20 | 2/8 | 0/8 |
| DAgger round 2 | 12/20 | 4/8 | 2/8 |
| DAgger round 3 | 9/20 | 6/8 | 3/8 |
| DAgger round 4 (last) | 11/20 | 5/8 | 0/8 |
| Shuffle mutant (dev) |  | 0/8, 0 invalid |  |
| Zero policy (dev) |  | 0/8, 8 timeouts, max displacement 0 |  |

Equal-budget BC stayed at 2/8 and 0/8, so the later DAgger gain is not explained by extra epochs on expert data. Always shipping round 4 missed the 0.25 OOD margin (0/8 versus BC 0/8). Selecting the checkpoint with highest **train** success (round 2: 12/20) ships 4/8 dev and 2/8 dev_ood, which meets `dagger_id >= bc_id` and `dagger_ood_rate >= bc_ood_rate + 0.25`. The planned BC floor of 4/8 was **not** achieved (2/8 on dev, 0/20 on train). Gates below are those prototype-backed values. Mutant gates are unchanged.

| Gate | Rule |
| --- | --- |
| Expert | Success 20/20 train, 8/8 id_test, 8/8 ood_test. Zero collisions. Zero `invalid`. |
| BC id | Closed-loop successes **>= 2**. |
| BC ood | `bc_ood_successes / 8 <= bc_id_rate - 0.25` (a 0.25 gap is 2 maps). At the measured 2/8 BC id floor this requires 0 OOD successes. Do not shrink this inequality, and do not edit locked test maps if a later locked run misses it. |
| DAgger | Measured on the **train-selected** checkpoint (Decision G). `dagger_ood_rate >= bc_ood_rate + 0.25` and `dagger_id_successes >= bc_id_successes`. |
| `label_shuffle` | Same BC budget (init, 40 epochs, same expert rows). One fixed permutation of **row indices** with `PCG64(20261111)` applied to the action pairs (the two components of a row stay together). Must finish 8 finite, valid, non-`invalid` id_test episodes. Successes **<= 1** and **strictly below** real BC id successes. A NaN output, exception, or crash is a hard CI failure, not a mutant pass. |
| `zero_policy` | Raw command is always `(0, 0)`, still through `apply_command`. 8 finite valid episodes, success **0**, and `max` hypot displacement from start **<= 1e-9** m. |

If the locked suites miss a narrative gate after a good-faith implementation, record the integer counts in `DEVIATIONS.md` and leave the tests failing until a plan amendment changes generator or training **without** loosening inequalities or editing test maps. Do not drop mutants or OOD.

### Decision J - Pages consume CI JSON

Schema version 1. The browser does not train and does not simulate. Vite + TypeScript + Canvas 2D. Base path `/nav-imitation-learning/`. Playwright, Chromium only, `npx playwright install --with-deps chromium`, `webServer` serving the built site, tests at `/nav-imitation-learning/`. `docs/demo.png` is produced once by a local Playwright run and committed. CI uploads a fresh screenshot artifact and does not commit it. `web/public/sample_report.json` has `"sample": true`. The CI report is copied to `web/public/report.json` **before** `vite build`. That file has `"sample": false`. The page must not fall back to the sample when the CI report is missing; it shows an error state. The HTML document title contains `simulation only` (for example `nav-imitation-learning dashboard (simulation only)`). Visible text, exact strings:

- `Simulation only`
- `Synthetic maps and a scripted expert`
- `Recorded CI output; this page does not train a policy.`

### Decision K - one workflow, pinned SHAs, determinism where it matters

`runs-on: ubuntu-24.04`. Never `ubuntu-latest`. Jobs `core`, `web` (`needs: [core]`), and `pages` (`needs: [core, web]`). `timeout-minutes: 25` on `core`; `15` on `web`; `15` on `pages`. Coverage **80%** line floor on authored modules `env`, `expert`, `policy`, `train`, `eval` only (omit `cli.py` and `web/`). `--cov-fail-under=80` must be able to fail. A negative coverage check uses a tiny deliberately uncovered fixture, not “raise the floor to 100.”

Same seeds ⇒ bit-identical occupancy bytes and demonstration tables (sha256 of a canonical dtype, C-order, documented endianness; not NPZ container bytes). Policy weights need not match across machines. Closed-loop success **counts** on the locked suites must be deterministic on the CI image with `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`, and `PYTHONHASHSEED=0`, all set **before** importing NumPy. Do not golden-compare wall-clock times or weight hashes across architectures.

### Decision L - honesty

No real robot logs, no Open X-Embodiment, no physical datasets, no customer sites, nothing tied to patent-pending work, no private or unreleased product algorithms. No claim of ISO or safety certification. No claim that this is Nav2, LeRobot, or a foundation model.

## Pinned stack

Verified 2026-11-11 against PyPI, npm, nodejs.org, and the GitHub API (Actions tag objects). Python runtime on the runner: **3.12** (latest 3.12.x that `setup-python` resolves; do not float to 3.13).

Direct Python packages, equality pins, **filenames and sha256** of the linux x86_64 (or py3-none-any) wheels:

| Package | Version | Wheel | sha256 |
| --- | --- | --- | --- |
| numpy | 2.5.4 | `numpy-2.5.4-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl` | `fbde6962867ee75b48b0ee29b2b9372ec5d617799dbaf38e82dc0596f2f7738a` |
| pytest | 9.2.0 | `pytest-9.2.0-py3-none-any.whl` | `07acdcf88ec174f907dc01af70bf52d49b014bbab03faeeade31c96b18c2e588` |
| pytest-cov | 7.1.0 | `pytest_cov-7.1.0-py3-none-any.whl` | `a0461110b7865f9a271aa1b51e516c9a95de9d696734a2f71e3e78f46e1d4678` |
| coverage | 7.17.2 | `coverage-7.17.2-cp312-cp312-manylinux1_x86_64.manylinux_2_28_x86_64.manylinux_2_5_x86_64.whl` | `36314997533266bde3d07765289674304d61bfe49165bb1b4fa9be12cb4ba0d1` |
| ruff | 0.17.4 | `ruff-0.17.4-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl` | `2cd025509b2e15721b426fc70fd946d71f43b3db3b08e946dfec2d696d911690` |
| mypy | 2.4.1 | `mypy-2.4.1-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl` | `a20ff6cd2af7a670c64f089bfa5fbcebe8ff3bf896c5e6ac7c9862c7960e4494` |
| hatchling | 1.33.0 | (build backend; pin in `pyproject.toml` and record the wheel hash at first lock) | verify at lock time |
| pip-tools | 7.6.2 | used to generate the lock; BSD-3-Clause | verify the wheel hash at lock time |

Do **not** pass `pip-compile --python-version`. Run pinned `pip-tools==7.6.2` **under Python 3.12**: `pip-compile --generate-hashes -o requirements.lock requirements.in`. The lock's transitive set is whatever that resolver writes; do not hand-edit transitive versions into this plan. CI installs with `pip install --require-hashes -r requirements.lock`, then installs the local package with `--no-deps` (after a wheel or editable install that does not bypass the hashed deps).

Node Active LTS: **26.13.0** Lithium, 2026-10-28, bundled npm **11.20.0** (https://nodejs.org/dist/index.json). npm packages, current latest on 2026-11-11, licenses checked on the registry:

| Package | Version | License |
| --- | --- | --- |
| vite | 8.3.8 | MIT |
| typescript | 7.0.2 | Apache-2.0 |
| vitest | 5.1.0 | MIT |
| @playwright/test | 1.65.0 | Apache-2.0 |
| @types/node | 26.6.6 | MIT |

Commit `package-lock.json` generated with npm 11.20.0. Playwright browsers: Chromium only, the build bundled with `@playwright/test@1.65.0`.

GitHub Actions, full commit SHAs (latest releases on 2026-11-11), never floating tags:

| Action | Tag | SHA |
| --- | --- | --- |
| actions/checkout | v7.0.1 | `3d3c42e5aac5ba805825da76410c181273ba90b1` |
| actions/setup-python | v7.1.0 | `6d4fd22b20465ccc0f7354476588e8df7c9ca0e0` |
| actions/setup-node | v7.1.0 | `949feb2413d6458794dcd2491c4babbbce0c15c1` |
| actions/upload-artifact | v7.0.2 | `cf430e030ddbb5b0abf93d22962f4752f3646cd9` |
| actions/download-artifact | v8.0.2 | `9000827ccba6bdab643e8b6fd33ac0654aef8333` |
| actions/configure-pages | v6.0.0 | `45bfe0192ca1faeb007ade9deae92b16b8254a0d` |
| actions/deploy-pages | v5.0.1 | `368f82528645a54fb793d4d04e342629a3f51346` |
| actions/upload-pages-artifact | v5.0.0 | `fc324d3547104276b827a68afc52ff2a11cc49c9` |

Sources: `https://pypi.org/pypi/<name>/<ver>/json`, `https://registry.npmjs.org/<name>/<ver>`, `https://nodejs.org/dist/index.json`, `https://api.github.com/repos/actions/<name>/git/refs/tags/<tag>`. The official Apache-2.0 text is 11358 bytes, sha256 `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`, from https://www.apache.org/licenses/LICENSE-2.0.txt.

The runner's exact 3.12 patch and the Ubuntu image float; that boundary is accepted. Fixture hashes must still match.

## File layout

Package `nav_imitation_learning/`: `constants.py`; `env/{plant,maps,collision,rays}.py`; `expert/{astar,tracker,rollout,label}.py`; `policy/mlp.py`; `train/{bc,dagger}.py`; `eval/{closed_loop,mutants,report}.py`; `cli.py`. Tests mirror those names plus `test_expert_oracle.py`, `test_gates.py`, `test_report_schema.py`, and `test_swept_collision.py`. Web: `web/index.html`, `web/src/{main,draw,schema}.ts`, `web/tests/dashboard.spec.ts`, `web/public/sample_report.json`. Also `fixtures/{train,id_test,ood_test}/` with sha256 sidecars (dev fixtures may be generated on demand and need not be committed), `docs/demo.png`, `.github/workflows/ci.yml`, `LICENSE`, `NOTICE`, `README.md`, `pyproject.toml`, `requirements.in`, `requirements.lock`.

SPDX-License-Identifier: Apache-2.0 on every authored Python and TypeScript file.

## Interfaces

### Python

```
step_unicycle(state, v, omega, dt) -> state
apply_command(v, omega, prev, dt) -> (v_applied, omega_applied, status)
clearance(pose, rectangles) -> float
swept_step(state, v_applied, omega_applied, dt, rectangles) -> (state, status)
plan_mask(occupancy, plan_inflation_m, res_m) -> lethal uint8
ray_ranges(occupancy, x, y, theta, res_m) -> float64[36]  # meters, pre-norm
pack_obs(ranges_m, goal_xy, pose, prev_cmd) -> float32[40]
astar(lethal, start_ij, goal_ij) -> list[(i, j)] | None
tracker_raw(pose, polyline, progress) -> (v_raw, omega_raw, progress)
label(world, pose, prev_applied, goal) -> ('ok', v, omega) | ('unlabelable', reason)
expert_episode(world, start, goal) -> Episode
mlp_init(rng) -> params
mlp_forward(params, obs_batch) -> (pre_clip[N,2], cmd[N,2])
adam_step(params, grads, moments, t) -> None
train_bc(params, rows, epochs, seed) -> params
train_dagger(init_params, worlds, bc_params) -> (params, rows)
eval_suite(policy, worlds) -> SuiteResult
write_report(path, payload) -> None   # JSON, allow_nan=False
```

### CLI

```
nil generate --out fixtures/
nil train-bc --fixtures fixtures/ --init-out artifacts/init.npz --out artifacts/bc.npz
nil train-dagger --fixtures fixtures/ --init artifacts/init.npz --bc artifacts/bc.npz --out artifacts/dagger.npz
# writes the train-selected checkpoint and records dagger_selected_round
nil eval --fixtures fixtures/ --bc artifacts/bc.npz --dagger artifacts/dagger.npz --out artifacts/report.json
nil report --check artifacts/report.json --require-ci --expect-sha $GITHUB_SHA
```

`nil eval` runs expert, BC, DAgger, and both mutants, and writes schema v1. Exit code 1 when any ship gate fails. `nil report --check --require-ci` recomputes every gate from episode records, rejects `sample: true`, rejects a zero SHA, rejects NaN, and exits 1 on a failed gate or a missing field.

### Report JSON (schema version 1)

Top-level: `schema_version` (1), `sample` (bool), `git_sha` (40 hex from `github.sha` / `GITHUB_SHA` passed into `nil eval` and `nil report --expect-sha`; never the literal `unknown`; 40 zeros only on the committed sample), `run_id` (string), `honesty` (exact: `Simulation only. Synthetic maps and a scripted expert. This file is recorded output; it is not a live training run.`), `seeds`, `constants` (`dt`, `goal_tol_xy`, `R_max`, `obs_dim`, `robot_radius_m`, `plan_inflation_m`), `open_loop_mse` (`bc_id`, `dagger_id`, `shuffle_id`, all finite numbers), `suites`, `episodes`, `trajectories`.

Each suite object: `n` (8, except no train row in this list), `successes`, `collisions`, `invalids`, `timeouts`, `success_rate`, `collision_rate`, `mean_time_success_s` (number or `null`), `expected` (`pass`|`fail`), `observed` (`pass`|`fail` **recomputed** by the writer from the rules in Decision I), `pass` (true only when `observed == expected`, also recomputed). Mutants set `expected` to `fail`. Keys are exactly the eight names in Decision H. A validator **recomputes** rates and gates from `episodes` and fails if the stored `pass` disagrees.

`episodes` holds one record per suite-seed (8×8 = 64 plus mutants' 16) with the metrics in Decision H, including `start`, `goal`, `final_pose`, and `status`.

Each trajectory (for drawing only): `suite`, `seed`, `status`, `start`, `goal`, `xy` (every 4th pose including the initial pose, plus the last if not already kept; at most 201 points, no duplicate trailing point), `obstacles` as `[[x,y,w,h], ...]`, `border` 0.2. Include all 8 id and all 8 ood episodes for expert, BC, and DAgger (48), plus the first id episode of each mutant.

### Pages

Route `/nav-imitation-learning/` loads `report.json` next to the built assets. Canvas draws rectangles, border, start, goal, and the polyline of the selected episode. A table lists the eight suites with integer counts and rates that **match the fetched JSON**. Sample mode shows `Sample report (not a CI run)` only when `sample === true`. Playwright asserts a locked seed's displayed counts, a canvas that is not merely a background fill (trajectory or obstacle pixels differ from the clear background), no console errors, the three honesty strings, a document title that contains `simulation only`, and the error state on a missing or 404 report without using a cached sample.

## Edge cases

- Reset already at the goal: `success`, `steps == 0`, no plant transition.
- Start or goal on a plan-lethal cell, or A* returning `None`: `invalid` if an episode is forced; the generator must not accept that pair.
- Collision and goal on the same step: `collision`.
- All-max-range scan: 36 ones. Legal.
- Yaw wrap is `(-π, π]`. A test with `θ = π - 0.01` and a target just past `-π` has `|e| ≤ π`. `wrap(-π) = π`.
- From rest, a request of `v_max` applies `0.05` on the first step.
- Non-finite command, pose, `dt`, or observation: `invalid`. A non-finite training batch raises and does not write NaN weights.
- A* refuses a diagonal when either orthogonal neighbor is plan-lethal (one-cell notch test).
- Swept grazing of a corner between two free endpoints is a collision (dedicated test).
- Two OOD box skips are allowed. Expert success is still required.
- `zero_policy` displacement is ~0. `mean_time_success_s` is `null` when successes are zero.
- Out-of-world successor: `invalid`.

## Test plan

Unit tests: unicycle quarter-circle and the 0.025 m straight step, small-angle sinc numbers in Verification-hints, wrap, rate limit, NaN, `dt <= 0`; continuous clearance (distance `< 0.20` collides, `== 0.20` does not) and a grazing-corner swept test; plan-mask conservatism; the box-edge ray `1.0 ± 1e-9` m, parallel rays, range 0 inside an occupied cell, diagonal-corner X-first, all quadrants, goal-frame rotation; A* octile equality on an empty grid, no corner cutting, insertion-index tie-break; tracker lookahead 0.50 m, `v_raw = 0` when `|e| > π/2`, already-at-goal raw zeros; Glorot bounds, nonzero spread, zero biases, forward shape `(N, 2)`, tanh, Adam one-step and a two-step moment persist, finite-difference grads, a toy MSE decrease, a non-finite batch. Fixture tests regenerate all committed families, match sha256 and box lists, check shape `(200, 200)`, and separations in `[3, 8]`.

The core job also runs the expert oracle (20/20, 8/8, 8/8, zero collisions, continuous contact), BC and DAgger gates recomputed from episode records, both mutants (healthy finite episodes plus the performance gates), finite open-loop MSE, schema checks (`sample: false`, exact honesty sentence, gate recomputation), `ruff check`, `mypy --strict` on the package, and `pytest --cov --cov-fail-under=80` on the locked module set.

Vitest plus a `tsc --noEmit` step. Playwright as in Decision J. The screenshot is an artifact.

## Numbered success criteria

1. `pytest tests/test_maps.py::test_fixture_hashes` - occupancy sha256 and box lists match the sidecar for every seed in `11001..11020`, `21001..21008`, `31001..31008`.
2. `pytest tests/test_expert_oracle.py` - expert success `20/20`, `8/8`, and `8/8`, collision count 0, `invalid` count 0, using continuous swept contact.
3. `pytest tests/test_gates.py::test_bc_id` - BC id successes `>= 2`.
4. `pytest tests/test_gates.py::test_bc_ood_gap` - `bc_ood_successes / 8 <= bc_id_rate - 0.25`.
5. `pytest tests/test_gates.py::test_dagger_margin` - on the train-selected DAgger checkpoint, dagger ood rate `>= bc_ood_rate + 0.25` and dagger id successes `>=` BC id successes.
6. `pytest tests/test_gates.py::test_mutant_shuffle` - 8 finite valid episodes, shuffle successes `<= 1` and `<` BC id successes. A crash or `invalid` fails this test.
7. `pytest tests/test_gates.py::test_mutant_zero` - 8 finite valid episodes, successes `== 0`, max displacement `<= 1e-9` m.
8. `pytest tests/test_plant.py::test_quarter_circle` - pose error `< 1e-12` against the analytic quarter circle. `test_small_angle` matches Verification-hints within `1e-12`.
9. `pytest tests/test_rays.py::test_box_edge` - range to the locked box is `1.0 ± 1e-9` m, not the cell-center distance.
10. `pytest tests/test_astar.py::test_no_corner_cut` and `tests/test_swept_collision.py::test_grazing_corner`.
11. `pytest tests/test_mlp.py::test_adam_one_step` and `test_grad_finite_diff`.
12. `pytest tests/test_report_schema.py` - CI report has `schema_version == 1`, `sample == false`, eight suites, 64+16 episode records, and every stored `pass` equals a recomputation from those records. Mutated reports that flip a count, a rate, `n`, or `sample` are rejected.
13. `nil report --check --require-ci` on a mutated report that sets `sample` true while claiming a real SHA exits 1 (`test_sample_flag_rejected_for_ci`).
14. `pytest --cov=nav_imitation_learning.env --cov=nav_imitation_learning.expert --cov=nav_imitation_learning.policy --cov=nav_imitation_learning.train --cov=nav_imitation_learning.eval --cov-fail-under=80` exits 0, and a deliberately uncovered tiny fixture with a raised floor exits non-zero.
15. `ruff check` and `mypy --strict` on `nav_imitation_learning` exit 0.
16. `npx tsc --noEmit`, `npx vitest run`, and `npx playwright test` exit 0. `dashboard.spec.ts` asserts the three visible strings, a document title containing `simulation only`, a locked seed's counts, and the missing-report error state.
17. `ci.yml`: `web` has `needs: [core]`; `pages` has `needs: [core, web]`, `if: github.ref == 'refs/heads/main' && github.event_name == 'push'`, downloads `web-dist`, and deploys only that tested tree.
18. README and NOTICE contain the honesty boundary and do not claim Nav2, LeRobot, or a foundation model. `tests/test_docs_honesty.py` greps the required phrases and forbids a short list of over-claim words (`certified`, `production driving policy`, `real robot logs`).
19. A CI `author-check` on the workflow SHA (`github.sha`) expects author and committer `Arpan Bom <applycoding@gmail.com>` and no co-author trailers.
20. `LICENSE` sha256 equals `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` (official 11358-byte text). CI step `license-bytes` fails on mismatch.

## Out of scope

ROS 2, Gazebo, Nav2, real lidars, rosbag playback, language-conditioned actions, PPO or any other RL optimizer, VLA models, CLIP, PyTorch, discrete action heads, dynamic obstacles, moving goals, heading-at-goal constraints, multi-robot planning, human teleop, in-browser training, model zoos, and any claim about on-robot transfer. Coulter's exact curvature law is out unless a later amendment replaces the heading tracker.

## Documentation and honesty

README: task, the 40-D observation, `nil` commands, the gate table, and the Pages URL. It states that maps and the expert are synthetic, that the tracker is a lookahead heading controller (not Coulter's law, not a Nav2 controller), that offline MSE is not closed-loop success, and that the aggregation loop is inspired by Ross, Gordon, and Bagnell 2011. NOTICE starts with `nav-imitation-learning, Copyright 2026 Arpan Bom` and lists **redistributed or installed** direct and dev dependencies with version and license, not merely idea citations. No real logs, no OXE, no customer sites, no patent-pending material, no safety certification.

## License

Outbound **Apache-2.0**. `LICENSE` is the verbatim official text (hash above). SPDX header on every authored source file.

Idea-only references, licences checked 2026-11-11, are listed in REFERENCES.md. Do not copy those projects. Do not copy applycoding/robot-learning-vla-demo (MIT). Academic citations: Ross, Gordon, Bagnell, AISTATS 2011 (https://proceedings.mlr.press/v15/ross11a.html, PDF https://proceedings.mlr.press/v15/ross11a/ross11a.pdf); Hart, Nilsson, and Raphael, 1968, for A*; Amanatides and Woo, Eurographics 1987, for the raycast *algorithm*; Coulter, CMU-RI-TR-92-01, 1992, as a related path-tracking *idea* only (https://www.ri.cmu.edu/pub_files/pub3/coulter_r_craig_1992_1/coulter_r_craig_1992_1.pdf, observed to redirect to the Robotics Institute publications host with HTTP 200 on 2026-11-11).

## Build and CI

Workflow `.github/workflows/ci.yml`. Triggers: `push` and `pull_request`. Permissions default to `contents: read`; the `pages` job adds `pages: write` and `id-token: write`. Concurrency: `pages` group without cancel-in-progress; PR jobs may cancel in progress. No `pull_request_target`. No secrets in fork PRs.

`core` (`ubuntu-24.04`, 25 minutes): checkout SHA pin; setup-python 3.12 SHA pin; set thread env vars; if `git rev-parse` is used, set `safe.directory` to the workspace so the SHA is not silently `unknown`; `pip install --require-hashes -r requirements.lock`; install the local package `--no-deps`; `nil generate` to a temp dir and diff hashes to `fixtures/`; `ruff`; `mypy --strict`; `nil train-bc`; `nil train-dagger`; `nil eval` writing `artifacts/report.json`; `nil report --check --require-ci --expect-sha ${{ github.sha }}`; `pytest` including oracles, gates (which may load the just-written artifacts rather than retraining), and coverage fail-under 80; `author-check`; `license-bytes`; a typographic-dash check that fails if U+2014, U+2013, or U+2212 appears in authored files (exclude lockfiles, `node_modules`, and generated coverage); upload artifact `report` (the JSON) and `models` (npz). A missing upload fails the job.

`web` (`needs: [core]`, 15 minutes): checkout; setup-node 26.13.0 with npm cache; `npm ci`; `npx tsc --noEmit`; `npx vitest run`; download `report`; copy to `web/public/report.json` (fail if missing); `npx vite build --base /nav-imitation-learning/`; `npx playwright install --with-deps chromium`; Playwright against the built site; upload screenshot artifact and `web-dist` (the tested `web/dist` that already contains `report.json`). Do not `git add` the screenshot.

`pages` (`needs: [core, web]`, 15 minutes): `if: github.ref == 'refs/heads/main' && github.event_name == 'push'`; environment `github-pages`; download `web-dist` into a known directory; configure-pages; upload-pages-artifact on that directory; deploy-pages. A failed main run must not replace a previously successful Pages deployment with a sample or failed report.

If a prototype of this full job exceeds 25 minutes on a GitHub-hosted runner, record the time and only then shrink DAgger epochs or hidden width. Do not delete OOD, DAgger, or mutants.

## Risks

- **Prototype-backed margins, with variance on n=8.** Decision I records the 2026-11-11 prototype. BC closed-loop success was 2/8 on dev and 0/20 on train; the 4/8 floor was dropped to **>= 2**. The 0.25 OOD gap and DAgger margin held for the train-selected checkpoint (round 2) and failed if round 4 was always shipped, which is why Decision G selects on train success. Locked id/ood lists were not used for this measurement and must not be edited to create the story. An 8-map gate can still miss on a fresh locked list; if that happens, record integer counts in `DEVIATIONS.md` rather than loosening inequalities.
- **Partial observability.** A reactive 40-D MLP may not represent a globally informed teacher. If the prototype cannot reach the margin, add only a documented, tiny memory feature (for example the previous two commands) in a plan amendment, or simplify **dev** topology first.
- **CI time.** The recorded prototype ran generate + BC + four DAgger rounds + evals in 46.1 s locally by caching a shortest-path tree per goal. Keep that cache; reuse trained artifacts in pytest. Do not shrink the 25-minute `core` timeout unless a full CI image run exceeds it.
- **Lookahead corner cutting** on 0.90 m OOD gaps. The plan-inflation margin and the expert safety overlay exist to protect the 100% expert oracle. If the overlay brakes too often, expert time-outs will fail the generator; fix the tracker, not the oracle.
- **Weights** are not bit-identical across BLAS. Hash maps, demos, and success counts.
- **Raycast ties and inflation vs pursuit.** The X-first corner rule and the 1.0 m box-edge test are merge blockers. Continuous contact is the ship oracle, not the planning mask.
- **Label rate limits.** Labels use the rollout's previous applied command. Label queries must not advance that state twice.
- **Sample fallback.** Production shows an error if `report.json` is absent.
- **mypy --strict** on numpy code may need explicit annotations; do not weaken the CI step, annotate instead.
- **Pin rot.** Pins are dated 2026-11-11. A refresh updates this plan and the lockfile together.

## Verification-hints

Seed for weight init, shuffle permutation, and training: **20261111**.

**Quarter circle.** `v = 1`, `ω = 1`, `dt = π/2`, start `(0, 0, 0)` ⇒ `(1, 1, π/2)` within `1e-12`. Also `v = 0.5`, `ω = 0`, `dt = 0.05`, `θ = 0` ⇒ `Δx = 0.025`, `Δy = 0`.

**Sinc at a small angle.** `v = 0.5`, `ω = 1.2`, `dt = 0.05`, start `(0, 0, 0)`:

```
dθ = 0.06
s  = sin(0.03) / 0.03 = 0.9998500067498554
Δx = 0.5 * 0.05 * s * cos(0.03) = 0.024985002699768586
Δy = 0.5 * 0.05 * s * sin(0.03) = 0.0007497750269982645
```

These replace an earlier incorrect printing of `Δx` and `Δy`. Tolerance `1e-12`.

**Rate limit.** From `(0, 0)`, requested `(0.60, 1.20)` ⇒ applied `(0.05, 0.15)`.

**Box ray.** Resolution 0.05. Occupied rectangle `[2.0, 3.0] × [0.0, 2.0]`. Sensor at `(1.0, 1.0)`, `θ = 0`, beam 0. Range **1.0 m**. Distance to that cell's center would be about `1.025` m and must not be the feature. Normalized feature `0.25`.

**Octile.** From cell `(0, 0)` to `(3, 1)` on an empty grid, optimal cost is `sqrt(2) + 2`. Heuristic at the start equals that cost.

**Glorot limits.** `40→64`: `sqrt(6/104) = 0.24019223070763071`. `64→64`: `sqrt(6/128) = 0.21650635094610965`. `64→2`: `sqrt(6/66) = 0.30151134457776363`. Test `|w| <= a` and a nonzero variance.

**Adam one step.** Scalar `w = 0`, `g = 1`, `t = 1`, moments start at 0, `lr = 3e-3`:

```
m = 0.1
v = 0.001
mhat = 1
vhat = 1
w' = 0 - 3e-3 * 1 / (sqrt(1) + 1e-8) = -0.0029999999700000006
```

**Gate arithmetic (`n = 8`).** A rate gap of 0.25 is **2 maps**. BC id at the prototype-backed floor is 2/8, then BC ood successes must be `<= 0`, and DAgger ood must be at least 2/8 when BC ood is 0. If BC id is 3/8, BC ood must be `<= 1/8`; if BC id is 4/8, BC ood must be `<= 2/8`. Shuffle allows at most 1/8 and must stay strictly below BC id. Zero policy allows 0/8. Expert allows 0 misses on 20 + 8 + 8 episodes. DAgger inequalities are applied to the train-selected checkpoint, not automatically to round 4.

**Demo size bound.** Twenty expert episodes of at most 800 steps ⇒ at most 16_000 demonstration rows before aggregation. Four rounds add at most another 64_000 rows. That bound is the CI cost to watch, especially A* relabelling.

**Feature index.** `obs[0]` is the ray at `-π`, `obs[35]` is the ray at `17π/18`, `obs[36]` is `gx / R_max`, `obs[37]` is `gy / R_max`, `obs[38]` is `v / v_max`, `obs[39]` is `ω / omega_max`.

Do not publish a gate change that lacks recorded prototype integer counts, the seed `20261111`, and a statement of whether **dev** or locked test maps were used.
