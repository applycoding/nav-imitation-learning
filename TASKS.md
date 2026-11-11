# Build tasks - nav-imitation-learning

Execute in order. Follow PLAN.md. Deviate only if a pin or a test proves a step broken, and record why in DEVIATIONS.md. License is **Apache-2.0**. Commit author and committer are always **Arpan Bom <applycoding@gmail.com>**. No co-author trailers. No credit to any assistant or tool in the repository.

Each task has acceptance checks. A task is done only when those checks pass. Do not loosen a gate, edit id_test or ood_test maps, or drop a mutant to make CI green.

## 1. Repository skeleton and license

- Create the tree in PLAN.md (empty dirs may hold `.gitkeep`).
- Fetch `https://www.apache.org/licenses/LICENSE-2.0.txt` into `LICENSE` and confirm sha256 `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` (11358 bytes).
- Write `NOTICE` starting with `nav-imitation-learning, Copyright 2026 Arpan Bom`, then dependency notices (filled as pins land).
- README stub: portfolio demo, simulation only, synthetic maps and a scripted expert, lookahead heading tracker (not Coulter's law, not Nav2), not a foundation model, not applycoding/robot-learning-vla-demo, Apache-2.0, author Arpan Bom.
- `pyproject.toml` with hatchling 1.33.0, package `nav_imitation_learning`, script `nil`. SPDX on authored files.
- `.gitignore`: `__pycache__/`, `.venv/`, `node_modules/`, `dist/`, `artifacts/`, `.pytest_cache/`, coverage files, local Playwright output. Fixtures are committed; local `dev` generations are ignored unless explicitly added.

**Accept:** `sha256sum LICENSE` matches the official digest. NOTICE and README name Apache-2.0 and Arpan Bom.

## 2. Lock Python and Node pins

- Write `requirements.in` with the direct pins in PLAN.md.
- Under Python 3.12, with `pip-tools==7.6.2`, run `pip-compile --generate-hashes -o requirements.lock requirements.in`. Do not pass a `--python-version` flag.
- Record hatchling and pip-tools wheel hashes in NOTICE once installed.
- `web/package.json` and `package-lock.json` with Node 26.13.0 / npm 11.20.0 and the npm pins in PLAN.md.

**Accept:** `pip install --require-hashes -r requirements.lock` succeeds on Python 3.12. The lock was produced by the pinned pip-tools.

## 3. Plant, clearance, and swept collision

- Implement `step_unicycle`, wrap, `apply_command`, point-AABB clearance, and swept-step sampling exactly as in PLAN.md.
- No wall clock. `dt` is an argument.

**Accept:** Criteria 8 and 10 (plant and grazing-corner). Rate-limit and NaN tests in the test plan.

## 4. Maps, planning mask, and generator

- Implement snapping, border, AABB gaps, the exact RNG order, start/goal sampling, and OOD extra-box rules.
- Planning mask uses `plan_inflation_m = 0.24`. Contact uses `robot_radius_m = 0.20` on original rectangles.
- Commit train, id_test, and ood_test fixtures plus sha256 sidecars. Do not commit tuned replacements for those seeds.

**Accept:** Criterion 1. Train gaps meet the 1.20 m rule. Every committed start/goal has separation in `[3, 8]` m.

## 5. Rays and observations

- Amanatides-Woo raycast with the half-open grid contract and X-first diagonal ties.
- `pack_obs` dimension 40, locked index order.

**Accept:** Criterion 9, plus quadrant, grid-line, and out-of-bounds tests in the test plan.

## 6. Expert: A* and lookahead heading tracker

- A* on the planning mask with the locked neighbor order and tie-breaks.
- Lookahead heading controller (0.50 m, `heading_kp = 1.5`), one `apply_command`, expert safety overlay.
- Stateless `label(...)` that does not mutate a live episode.

**Accept:** Criteria 2 and 10 (A* notch). Expert 100% uses continuous swept contact.

## 7. MLP, BC, and DAgger

- `40 -> 64 -> 64 -> 2`, Glorot init, Adam, locked loss, 40-epoch BC, four aggregation rounds of 20 epochs from the saved Xavier weights.
- After BC and each round, score closed-loop success on the 20 train maps and ship the best checkpoint (ties: fewer train collisions, then later round). Record `dagger_selected_round`.
- Artifacts are float64 npz files. Mutant shuffle uses the same budget and a fixed row-index permutation.

**Accept:** Criterion 11. `nil train-bc` and `nil train-dagger` write the artifacts used by eval.

## 8. Eval, mutants, report writer

- Episode loop order in Decision H.
- Both mutants must produce 8 finite valid episodes; crashes and `invalid` are CI failures, not mutant passes.
- `write_report` uses `allow_nan=False`. `nil report --check --require-ci` recomputes every gate from episode records.

**Accept:** Criteria 3-7 and 12-13.

## 9. Pages dashboard and Playwright

- Vite app, `base: '/nav-imitation-learning/'`, Canvas 2D, honesty labels, fetches via `import.meta.env.BASE_URL`.
- Copy CI `report.json` into `web/public/` before the production build. No silent sample fallback.
- Playwright as locked (Chromium with deps, `webServer` on the built site, 127.0.0.1, counts match the JSON, document title contains `simulation only`).
- Check in `docs/demo.png` locally once. CI uploads the screenshot and does not commit it.

**Accept:** Criterion 16.

## 10. Single workflow, honesty, authorship

- One `ci.yml` as in PLAN.md: `core`, `web` needs `core`, `pages` needs both and deploys only a main push of that SHA. SHA-pinned actions, Python 3.12, Node 26.13.0, `ubuntu-24.04`. Never `ubuntu-latest`.
- Thread-limit environment variables set before NumPy is imported.
- Local package installed with `--no-deps` after hashed requirements.
- `author-check` and `license-bytes`. Typographic-dash check (U+2014, U+2013, U+2212). README honesty test. Pass `GITHUB_SHA` into the report so `git_sha` is never `unknown`.

**Accept:** Criteria 14, 15, 17-20. CI is green and the final run on the pushed branch has **completed**.
