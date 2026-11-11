# SPDX-License-Identifier: Apache-2.0
"""Synthetic occupancy map generation and planning mask."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt

from nav_imitation_learning.constants import (
    BORDER_M,
    DEV_OOD_SEEDS,
    DEV_SEEDS,
    ID_TEST_SEEDS,
    MAP_CELLS,
    MAP_RES_M,
    MAP_SIZE_M,
    OOD_TEST_SEEDS,
    PLAN_INFLATION_M,
    TRAIN_SEEDS,
)
from nav_imitation_learning.env.collision import Rect, clearance, rectangles_from_boxes
from nav_imitation_learning.env.plant import Pose, wrap

Family = Literal["train", "id_test", "ood_test", "dev", "dev_ood"]


def snap(v: float) -> float:
    return math.floor(v / MAP_RES_M + 1e-12) * MAP_RES_M


def _paint_rect(grid: npt.NDArray[np.uint8], x: float, y: float, w: float, h: float) -> None:
    i0 = int(round(x / MAP_RES_M))
    j0 = int(round(y / MAP_RES_M))
    i1 = int(round((x + w) / MAP_RES_M))
    j1 = int(round((y + h) / MAP_RES_M))
    i0 = max(0, min(MAP_CELLS, i0))
    j0 = max(0, min(MAP_CELLS, j0))
    i1 = max(0, min(MAP_CELLS, i1))
    j1 = max(0, min(MAP_CELLS, j1))
    grid[i0:i1, j0:j1] = 1


def _boxes_from_grid(grid: npt.NDArray[np.uint8]) -> list[list[float]]:
    """Reconstruct obstacle boxes (excluding border) from occupancy for export."""
    rects: list[Rect] = []
    visited = np.zeros_like(grid, dtype=bool)
    for i in range(MAP_CELLS):
        for j in range(MAP_CELLS):
            if grid[i, j] == 0 or visited[i, j]:
                continue
            if i < 4 or i >= MAP_CELLS - 4 or j < 4 or j >= MAP_CELLS - 4:
                continue
            x = i * MAP_RES_M
            y = j * MAP_RES_M
            w = MAP_RES_M
            h = MAP_RES_M
            rects.append((x, y, w, h))
            visited[i, j] = True
    merged: list[list[float]] = []
    for r in rects:
        merged.append([r[0], r[1], r[2], r[3]])
    return merged


def aabb_gap(a: Rect, b: Rect) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    a_x2, a_y2 = ax + aw, ay + ah
    b_x2, b_y2 = bx + bw, by + bh
    overlap_x = ax < b_x2 and bx < a_x2
    overlap_y = ay < b_y2 and by < a_y2
    if overlap_x and overlap_y:
        return 0.0
    dx = max(0.0, ax - b_x2, bx - a_x2)
    dy = max(0.0, ay - b_y2, by - a_y2)
    if overlap_x:
        return dy
    if overlap_y:
        return dx
    return math.hypot(dx, dy)


def border_gap(rect: Rect) -> float:
    x, y, w, h = rect
    inner = BORDER_M
    inner_max = MAP_SIZE_M - BORDER_M
    gaps = [
        x - inner,
        inner_max - (x + w),
        y - inner,
        inner_max - (y + h),
    ]
    return min(gaps)


def occupancy_sha256(occupancy: npt.NDArray[np.uint8]) -> str:
    return hashlib.sha256(occupancy.tobytes(order="C")).hexdigest()


def plan_mask(occupancy: npt.NDArray[np.uint8], plan_inflation_m: float, res_m: float) -> npt.NDArray[np.uint8]:
    """Legacy occupancy-based mask (tests only). Prefer plan_mask_from_rects."""
    rects = _occupancy_rectangles(occupancy)
    return plan_mask_from_rects(rects, plan_inflation_m, res_m)


def plan_mask_from_rects(rects: list[Rect], plan_inflation_m: float, res_m: float) -> npt.NDArray[np.uint8]:
    """Cell is plan-lethal when center distance to obstacle rects < plan_inflation_m."""
    lethal = np.zeros((MAP_CELLS, MAP_CELLS), dtype=np.uint8)
    border = [
        (0.0, 0.0, MAP_SIZE_M, BORDER_M),
        (0.0, MAP_SIZE_M - BORDER_M, MAP_SIZE_M, BORDER_M),
        (0.0, 0.0, BORDER_M, MAP_SIZE_M),
        (MAP_SIZE_M - BORDER_M, 0.0, BORDER_M, MAP_SIZE_M),
    ]
    all_rects = list(rects) + border
    for i in range(MAP_CELLS):
        for j in range(MAP_CELLS):
            cx = (i + 0.5) * res_m
            cy = (j + 0.5) * res_m
            min_d = min(_point_aabb_dist(cx, cy, r) for r in all_rects)
            if min_d < plan_inflation_m:
                lethal[i, j] = 1
    return lethal


def _point_aabb_dist(px: float, py: float, rect: Rect) -> float:
    x, y, w, h = rect
    cx = max(x, min(px, x + w))
    cy = max(y, min(py, y + h))
    return math.hypot(px - cx, py - cy)


def _point_rect_center_dist(px: float, py: float, rx: float, ry: float, rw: float, rh: float) -> float:
    cx = max(rx, min(px, rx + rw))
    cy = max(ry, min(py, ry + rh))
    return math.hypot(px - cx, py - cy)


def _occupancy_rectangles(occupancy: npt.NDArray[np.uint8]) -> list[Rect]:
    rects: list[Rect] = []
    for i in range(MAP_CELLS):
        for j in range(MAP_CELLS):
            if occupancy[i, j]:
                rects.append((i * MAP_RES_M, j * MAP_RES_M, MAP_RES_M, MAP_RES_M))
    return rects


@dataclass
class World:
    seed: int
    family: Family
    occupancy: npt.NDArray[np.uint8]
    boxes: list[list[float]]
    start: Pose
    goal: tuple[float, float]
    occupancy_sha256: str
    base_box_skips: int = 0
    extra_box_skips: int = 0
    lethal: npt.NDArray[np.uint8] = field(default_factory=lambda: np.zeros((1, 1), dtype=np.uint8))
    rectangles: list[Rect] = field(default_factory=list)

    def __post_init__(self) -> None:
        box_rects = rectangles_from_boxes(self.boxes)
        self.lethal = plan_mask_from_rects(box_rects, PLAN_INFLATION_M, MAP_RES_M)
        self.rectangles = box_rects


def family_for_seed(seed: int) -> Family:
    if seed in TRAIN_SEEDS:
        return "train"
    if seed in ID_TEST_SEEDS:
        return "id_test"
    if seed in OOD_TEST_SEEDS:
        return "ood_test"
    if seed in DEV_SEEDS:
        return "dev"
    if seed in DEV_OOD_SEEDS:
        return "dev_ood"
    raise ValueError(f"unknown seed {seed}")


def min_gap_for_family(family: Family) -> float:
    if family in ("ood_test", "dev_ood"):
        return 0.90
    return 1.20


def generate_map(
    seed: int,
    family: Family | None = None,
    *,
    skip_expert_check: bool = False,
) -> World | None:
    """Generate a map fixture; returns None if generation fails."""
    from nav_imitation_learning.expert.astar import astar
    from nav_imitation_learning.expert.rollout import expert_episode

    if family is None:
        family = family_for_seed(seed)
    min_gap = min_gap_for_family(family)
    rng = np.random.Generator(np.random.PCG64(seed))

    grid = np.zeros((MAP_CELLS, MAP_CELLS), dtype=np.uint8)
    _paint_border(grid)
    accepted_boxes: list[Rect] = []
    base_box_skips = 0
    n_boxes = int(rng.integers(5, 9))
    for _k in range(n_boxes):
        placed = False
        for _attempt in range(80):
            w = 0.50 + float(rng.random()) * 2.00
            h = 0.50 + float(rng.random()) * 2.00
            max_x = max(0.0, MAP_SIZE_M - BORDER_M - w - BORDER_M)
            max_y = max(0.0, MAP_SIZE_M - BORDER_M - h - BORDER_M)
            x = BORDER_M + float(rng.random()) * max_x
            y = BORDER_M + float(rng.random()) * max_y
            x, y, w, h = snap(x), snap(y), snap(w), snap(h)
            if w < 0.50 or h < 0.50:
                continue
            rect: Rect = (x, y, w, h)
            ok = True
            for other in accepted_boxes:
                if aabb_gap(rect, other) < min_gap:
                    ok = False
                    break
            if ok and border_gap(rect) < min_gap:
                ok = False
            if ok:
                accepted_boxes.append(rect)
                _paint_rect(grid, x, y, w, h)
                placed = True
                break
        if not placed:
            base_box_skips += 1

    box_rects = list(accepted_boxes)
    lethal = plan_mask_from_rects(box_rects, PLAN_INFLATION_M, MAP_RES_M)
    free_cells: list[tuple[float, float, int, int]] = []
    rects = box_rects
    for i in range(MAP_CELLS):
        for j in range(MAP_CELLS):
            if lethal[i, j]:
                continue
            cx = (i + 0.5) * MAP_RES_M
            cy = (j + 0.5) * MAP_RES_M
            if clearance((cx, cy, 0.0), rects) >= 0.0:
                free_cells.append((cx, cy, i, j))
    if len(free_cells) < 16:
        return None

    start: Pose | None = None
    goal: tuple[float, float] | None = None
    for _ in range(50):
        i0 = int(rng.integers(0, len(free_cells)))
        i1 = int(rng.integers(0, len(free_cells)))
        sx, sy, si, sj = free_cells[i0]
        gx, gy, gi, gj = free_cells[i1]
        yaw = wrap(float(rng.uniform(-math.pi, math.pi)))
        if math.hypot(gx - sx, gy - sy) < 3.0 or math.hypot(gx - sx, gy - sy) > 8.0:
            continue
        if astar(lethal, (si, sj), (gi, gj)) is None:
            continue
        world_tmp = World(
            seed=seed,
            family=family,
            occupancy=grid.copy(),
            boxes=[[b[0], b[1], b[2], b[3]] for b in accepted_boxes],
            start=(sx, sy, yaw),
            goal=(gx, gy),
            occupancy_sha256=occupancy_sha256(grid),
            base_box_skips=base_box_skips,
        )
        if not skip_expert_check:
            ep = expert_episode(world_tmp, world_tmp.start, world_tmp.goal)
            if ep["status"] != "success":
                continue
        start = (sx, sy, yaw)
        goal = (gx, gy)
        break
    if start is None or goal is None:
        return None

    extra_box_skips = 0
    if family in ("ood_test", "dev_ood"):
        grid, accepted_boxes, extra_box_skips = _add_ood_boxes(
            grid, accepted_boxes, lethal, start, goal, rng
        )

    boxes_out = [[r[0], r[1], r[2], r[3]] for r in accepted_boxes]
    occ_hash = occupancy_sha256(grid)
    world = World(
        seed=seed,
        family=family,
        occupancy=grid,
        boxes=boxes_out,
        start=start,
        goal=goal,
        occupancy_sha256=occ_hash,
        base_box_skips=base_box_skips,
        extra_box_skips=extra_box_skips,
    )
    if not skip_expert_check:
        ep = expert_episode(world, start, goal)
        if ep["status"] != "success":
            return None
    return world


def _paint_border(grid: npt.NDArray[np.uint8]) -> None:
    for i in range(MAP_CELLS):
        for j in range(MAP_CELLS):
            if i < 4 or i >= MAP_CELLS - 4 or j < 4 or j >= MAP_CELLS - 4:
                grid[i, j] = 1


def _add_ood_boxes(
    grid: npt.NDArray[np.uint8],
    boxes: list[Rect],
    lethal: npt.NDArray[np.uint8],
    start: Pose,
    goal: tuple[float, float],
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.uint8], list[Rect], int]:
    from nav_imitation_learning.expert.astar import astar, polyline_cell_centers
    from nav_imitation_learning.expert.rollout import expert_episode

    extra_skips = 0
    si = int(start[0] / MAP_RES_M)
    sj = int(start[1] / MAP_RES_M)
    gi = int(goal[0] / MAP_RES_M)
    gj = int(goal[1] / MAP_RES_M)
    path = astar(lethal, (si, sj), (gi, gj))
    if path is None:
        return grid, boxes, 2
    poly = polyline_cell_centers(path, start, goal, lethal)
    path_cells_set = set(path)

    for _box_idx in range(2):
        placed = False
        for _ in range(50):
            t = float(rng.random())
            s_off = (float(rng.random()) - 0.5) * 0.80
            pt, tangent = _point_on_polyline(poly, t)
            nx, ny = -tangent[1], tangent[0]
            norm = math.hypot(nx, ny)
            if norm < 1e-12:
                continue
            nx /= norm
            ny /= norm
            cx = pt[0] + nx * s_off
            cy = pt[1] + ny * s_off
            bw, bh = 0.60, 0.60
            x = snap(cx - bw / 2)
            y = snap(cy - bh / 2)
            x = max(BORDER_M, min(x, MAP_SIZE_M - BORDER_M - bw))
            y = max(BORDER_M, min(y, MAP_SIZE_M - BORDER_M - bh))
            rect: Rect = (x, y, bw, bh)
            grid_backup = grid.copy()
            _paint_rect(grid, x, y, bw, bh)
            lethal_new = plan_mask_from_rects(boxes + [rect], PLAN_INFLATION_M, MAP_RES_M)
            box_cells = _rect_cells(rect)
            hits_path = any(c in path_cells_set for c in box_cells)
            if not hits_path:
                grid = grid_backup
                continue
            if astar(lethal_new, (si, sj), (gi, gj)) is None:
                grid = grid_backup
                continue
            w = World(
                seed=0,
                family="dev_ood",
                occupancy=grid,
                boxes=[[b[0], b[1], b[2], b[3]] for b in boxes] + [[x, y, bw, bh]],
                start=start,
                goal=goal,
                occupancy_sha256="",
            )
            ep = expert_episode(w, start, goal)
            if ep["status"] == "success":
                boxes = boxes + [rect]
                lethal = lethal_new
                placed = True
                break
            grid = grid_backup
        if not placed:
            extra_skips += 1
    return grid, boxes, extra_skips


def _rect_cells(rect: Rect) -> list[tuple[int, int]]:
    x, y, w, h = rect
    i0 = int(round(x / MAP_RES_M))
    j0 = int(round(y / MAP_RES_M))
    i1 = int(round((x + w) / MAP_RES_M))
    j1 = int(round((y + h) / MAP_RES_M))
    return [(i, j) for i in range(i0, i1) for j in range(j0, j1)]


def _point_on_polyline(
    poly: list[tuple[float, float]], t: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    if len(poly) < 2:
        return poly[0], (1.0, 0.0)
    seg_lens = [math.hypot(poly[i + 1][0] - poly[i][0], poly[i + 1][1] - poly[i][1]) for i in range(len(poly) - 1)]
    total = sum(seg_lens)
    if total < 1e-12:
        return poly[0], (1.0, 0.0)
    target = t * total
    acc = 0.0
    for i, slen in enumerate(seg_lens):
        if acc + slen >= target or i == len(seg_lens) - 1:
            local_t = (target - acc) / slen if slen > 1e-12 else 0.0
            local_t = max(0.0, min(1.0, local_t))
            px = poly[i][0] + local_t * (poly[i + 1][0] - poly[i][0])
            py = poly[i][1] + local_t * (poly[i + 1][1] - poly[i][1])
            tx = poly[i + 1][0] - poly[i][0]
            ty = poly[i + 1][1] - poly[i][1]
            return (px, py), (tx, ty)
        acc += slen
    return poly[-1], (1.0, 0.0)


def world_from_fixture(path: Path) -> World:
    meta_path = path / "meta.json"
    occ_path = path / "occupancy.npy"
    data = json.loads(meta_path.read_text())
    occ = np.load(occ_path)
    return World(
        seed=data["seed"],
        family=data["family"],
        occupancy=occ,
        boxes=data["boxes"],
        start=(data["start_xytheta"][0], data["start_xytheta"][1], data["start_xytheta"][2]),
        goal=(data["goal_xy"][0], data["goal_xy"][1]),
        occupancy_sha256=data["occupancy_sha256"],
        base_box_skips=data.get("base_box_skips", 0),
        extra_box_skips=data.get("extra_box_skips", 0),
    )


def write_fixture(world: World, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "occupancy.npy", world.occupancy)
    meta = {
        "seed": world.seed,
        "family": world.family,
        "start_xytheta": list(world.start),
        "goal_xy": list(world.goal),
        "occupancy_sha256": world.occupancy_sha256,
        "boxes": world.boxes,
        "base_box_skips": world.base_box_skips,
        "extra_box_skips": world.extra_box_skips,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (out_dir / "occupancy.sha256").write_text(world.occupancy_sha256 + "\n")


def load_fixtures(fixtures_root: Path) -> list[World]:
    worlds: list[World] = []
    for family in ("train", "id_test", "ood_test"):
        fam_dir = fixtures_root / family
        if not fam_dir.is_dir():
            continue
        for child in sorted(fam_dir.iterdir()):
            if child.is_dir() and (child / "meta.json").exists():
                worlds.append(world_from_fixture(child))
    return worlds
