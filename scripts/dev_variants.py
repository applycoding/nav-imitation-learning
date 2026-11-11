# SPDX-License-Identifier: Apache-2.0
"""Dev-only training-variant sweep (PLAN.md Amendment 1).

Trains on the committed train fixtures and scores train, dev (41001..41008) and
dev_ood (42001..42008). It never loads id_test or ood_test.

Usage: python scripts/dev_variants.py NAME '{"rounds": 8}'
"""

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"

import copy  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from nav_imitation_learning.constants import (  # noqa: E402
    DEV_OOD_SEEDS,
    DEV_SEEDS,
    DT,
    GOAL_TOL_XY,
    OBS_DIM,
    OMEGA_MAX,
    T_MAX_STEPS,
    V_MAX,
    WEIGHT_SEED,
)
from nav_imitation_learning.env.collision import clearance, plant_step  # noqa: E402
from nav_imitation_learning.env.maps import generate_map, load_fixtures  # noqa: E402
from nav_imitation_learning.env.rays import observe  # noqa: E402
from nav_imitation_learning.eval.closed_loop import policy_episode  # noqa: E402
from nav_imitation_learning.expert.label import label  # noqa: E402
from nav_imitation_learning.policy.mlp import glorot_limit, mlp_forward  # noqa: E402
from nav_imitation_learning.train.bc import train_bc  # noqa: E402
from nav_imitation_learning.train.demos import collect_all_train_demos  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "artifacts" / "dev_variants"
CACHE = OUT / "dev_worlds.pkl"


def dev_worlds():
    OUT.mkdir(parents=True, exist_ok=True)
    if CACHE.exists():
        return pickle.loads(CACHE.read_bytes())
    d = [generate_map(s, "dev") for s in DEV_SEEDS]
    do = [generate_map(s, "dev_ood") for s in DEV_OOD_SEEDS]
    assert all(w is not None for w in d + do)
    CACHE.write_bytes(pickle.dumps((d, do)))
    return d, do


def init(hidden, seed=WEIGHT_SEED):
    rng = np.random.Generator(np.random.PCG64(seed))
    p = {}
    for (fi, fo), nw, nb in zip(
        [(OBS_DIM, hidden), (hidden, hidden), (hidden, 2)], ["W1", "W2", "W3"], ["b1", "b2", "b3"], strict=True
    ):
        a = glorot_limit(fi, fo)
        p[nw] = rng.uniform(-a, a, size=(fi, fo))
        p[nb] = np.zeros(fo)
    return p


def rollout_rows(world, params, beta, rng):
    obs_rows, act_rows = [], []
    pose = world.start
    prev = (0.0, 0.0)
    goal = world.goal
    for _ in range(T_MAX_STEPS):
        if math.hypot(goal[0] - pose[0], goal[1] - pose[1]) <= GOAL_TOL_XY:
            break
        if clearance(pose, world.rectangles) < 0.0:
            break
        o = observe(world.occupancy, pose, goal, prev)
        if not np.all(np.isfinite(o)):
            break
        lab = label(world, pose, prev, goal)
        if lab[0] == "ok":
            obs_rows.append(o.astype(np.float64))
            act_rows.append([lab[1] / V_MAX, lab[2] / OMEGA_MAX])
        if beta > 0 and lab[0] == "ok" and rng.random() < beta:
            v, w = lab[1], lab[2]
        else:
            _, cmd = mlp_forward(params, o.astype(np.float64).reshape(1, -1))
            v, w = float(cmd[0, 0]), float(cmd[0, 1])
        pose2, prev2, st = plant_step(pose, v, w, prev, DT, world.rectangles)
        if st != "ok":
            break
        pose, prev = pose2, prev2
    if not obs_rows:
        return np.zeros((0, 40)), np.zeros((0, 2))
    return np.stack(obs_rows), np.array(act_rows)


def score(params, worlds):
    s = c = t = 0
    for w in worlds:
        ep = policy_episode(w, params, w.start, w.goal)
        s += ep["status"] == "success"
        c += bool(ep["collision"])
        t += ep["status"] == "timeout"
    return s, c, t


def run(
    name, rounds=4, bc_epochs=40, dag_epochs=20, hidden=64, beta0=0.0, beta_decay=0.5, warm=False, seed=WEIGHT_SEED
):
    t0 = time.time()
    worlds = load_fixtures(REPO / "fixtures")
    train = [w for w in worlds if w.family == "train"]
    dev, dev_ood = dev_worlds()
    obs, act = collect_all_train_demos(train)
    xav = init(hidden, seed)
    cur = train_bc(copy.deepcopy(xav), obs, act, epochs=bc_epochs, seed=seed)
    ckpts = [(0, copy.deepcopy(cur))]
    ao, aa = [obs], [act]
    rng = np.random.Generator(np.random.PCG64(WEIGHT_SEED + 7))
    for r in range(1, rounds + 1):
        beta = beta0 * (beta_decay ** (r - 1)) if beta0 > 0 else 0.0
        ro, ra = [], []
        for w in train:
            o, a = rollout_rows(w, cur, beta, rng)
            if o.size:
                ro.append(o)
                ra.append(a)
        if ro:
            ao.append(np.vstack(ro))
            aa.append(np.vstack(ra))
        start = copy.deepcopy(cur) if warm else copy.deepcopy(xav)
        cur = train_bc(start, np.vstack(ao), np.vstack(aa), epochs=dag_epochs, seed=seed)
        ckpts.append((r, copy.deepcopy(cur)))
    rows = []
    for r, p in ckpts:
        rows.append({"round": r, "train": score(p, train), "dev": score(p, dev), "dev_ood": score(p, dev_ood)})
    sel = sorted(rows, key=lambda x: (x["train"][0], -x["train"][1], x["round"]), reverse=True)[0]
    out = {
        "name": name,
        "cfg": dict(
            rounds=rounds,
            bc_epochs=bc_epochs,
            dag_epochs=dag_epochs,
            hidden=hidden,
            beta0=beta0,
            beta_decay=beta_decay,
            warm=warm,
            seed=seed,
        ),
        "rows": rows,
        "train_selected_round": sel["round"],
        "secs": round(time.time() - t0, 1),
    }
    print(json.dumps(out))
    with open(OUT / "results.jsonl", "a") as f:
        f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    cfg = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    run(sys.argv[1], **cfg)
