# nav-imitation-learning

Portfolio demo by **Arpan Bom** ([applycoding@gmail.com](mailto:applycoding@gmail.com)).
CPU-only NumPy **simulation** lab for imitation learning in planar navigation on small synthetic maps.

**Simulation only.** Synthetic maps and a scripted expert. This repository does not
collect real robot logs, does not run on-robot hardware, and is not a production
driving policy. It is not a foundation vision-language-action model, not Nav2, and
not [LeRobot](https://github.com/huggingface/lerobot). It is not
[applycoding/robot-learning-vla-demo](https://github.com/applycoding/robot-learning-vla-demo)
(that project is language-conditioned 2-link arm reaching with PyTorch).

## Task

A kinematic differential-drive robot navigates synthetic 10 m x 10 m occupancy maps.
A scripted expert (8-connected A* on a planning mask plus a **lookahead heading
tracker**) records demonstrations. A NumPy MLP (`40 -> 64 -> 64 -> 2`) is trained
with behavior cloning, then four DAgger-style aggregation rounds relabel
policy-visited states. Closed-loop evaluation on held-out maps checks covariate
shift; open-loop MSE on frozen expert states is a regression check only, not a
substitute for closed-loop success.

The tracker is inspired by path-tracking ideas but is **not** Coulter's curvature
law and is **not** a Nav2 controller.

### Observation (40-D)

1. Thirty-six lidar-like ranges (Amanatides-Woo on the original grid), normalized
   by `R_max = 4.0` m.
2. Goal in the robot frame (`gx`, `gy`), clipped and normalized.
3. Last applied command (`v / v_max`, `omega / omega_max`).

## CLI (`nil`)

Install: `pip install --require-hashes -r requirements.lock` then `pip install
--no-deps -e .`

| Command | Purpose |
| --- | --- |
| `nil generate --out fixtures/` | Regenerate committed train, id_test, ood_test fixtures |
| `nil train-bc --fixtures ... --init-out ... --out ...` | Xavier init plus 40-epoch BC |
| `nil train-dagger --fixtures ... --init ... --bc ... --out ...` | Four aggregation rounds; writes train-selected checkpoint |
| `nil eval --fixtures ... --bc ... --dagger ... --out ...` | Expert, BC, DAgger, mutants; schema v1 JSON |
| `nil report --check artifacts/report.json --require-ci --expect-sha $GITHUB_SHA` | Recompute gates from episode records |

## Ship gates (integer counts, not weakened in tests)

| Gate | Rule |
| --- | --- |
| Expert | 20/20 train, 8/8 id_test, 8/8 ood_test; zero collisions and invalid |
| BC id | Closed-loop successes >= 2 on id_test |
| BC ood | `bc_ood_rate <= bc_id_rate - 0.25` |
| DAgger | Train-selected checkpoint: `dagger_id >= bc_id`, `dagger_ood >= bc_ood`, and DAgger OOD collisions `<=` BC OOD collisions (amended; see below) |
| Shuffle mutant | 8 finite id episodes; successes <= 1 and < BC id |
| Zero policy | 8 finite episodes; 0 successes; max displacement <= 1e-9 m |

## Results (locked test maps, seed 20261111)

| Policy | id_test successes | ood_test successes | ood_test collisions |
| --- | --- | --- | --- |
| Expert (A* planner plus tracker, the baseline) | 8/8 | 8/8 | 0 |
| Behavior cloning | 3/8 | 0/8 | 8 |
| DAgger, train-selected round 3 | 4/8 | 1/8 | 7 |

What this shows, plainly:

- In distribution, DAgger reached 4/8 against 3/8 for BC on id_test (6/8 against 2/8 on the dev maps), still well below the 8/8 planner baseline.
- Out of distribution, both learned policies mostly collide. DAgger 1/8 against BC 0/8 is one map, and with only 8 OOD episodes a DAgger advantage over BC out of distribution is **not demonstrated**.
- The plan originally required DAgger to beat BC on ood_test by 2 maps. That gate was missed and was amended to a no-regression gate after training changes tried on dev maps did not help. The counts and every variant tried are in `DEVIATIONS.md` and PLAN.md Amendment 1.
- Both mutants (shuffled labels, zero commands) run 8 healthy episodes and fail their gates as intended (0/8 each).

Recorded CI output for a passing main-branch run is shown on GitHub Pages at
`https://applycoding.github.io/nav-imitation-learning/`. The page **does not train**
and **does not simulate**; it only renders `report.json` from CI.

## Dashboard (Vite + TypeScript)

```bash
cd web && npm ci
cp ../artifacts/report.json public/report.json   # after a local eval
npm run build
npx playwright test
```

`web/public/sample_report.json` has `"sample": true` for local smoke. Production
builds embed the CI `report.json` copied before `vite build`.

## Tests

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0
pytest tests/ --ignore=tests/cov_uncovered_stub \
  --cov=nav_imitation_learning.env --cov=nav_imitation_learning.expert \
  --cov=nav_imitation_learning.policy --cov=nav_imitation_learning.train \
  --cov=nav_imitation_learning.eval --cov-fail-under=80
```

Set `ARTIFACTS_DIR` to point gate tests at a fresh `nil eval` output tree.

See `DEVIATIONS.md` for the locked-suite integers, the missed DAgger OOD margin, and the amended gate.

## Author

Arpan Bom <applycoding@gmail.com>

## License

[Apache License, Version 2.0](LICENSE). SPDX headers on authored sources.
