from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple

import numpy as np

from abstraction import Rect
from main import (
    X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX,
    Y_GOAL, Y_OBS, R_GOAL, R_OBS,
    unicycle_dynamics,
)


GT_CACHE_VERSION = 1
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = SCRIPT_DIR / "artifacts" / "cache"


def default_gt_cache_path(nx_gt: int, ny_gt: int, nz_gt: int, max_steps: int) -> Path:
    return DEFAULT_CACHE_DIR / (
        f"gt_safe_unicycle_{nx_gt}x{ny_gt}x{nz_gt}_steps{max_steps}.pkl"
    )


def _rect_domain() -> Rect:
    return Rect(X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX)


def corners_are_safe(
    x_lo: float,
    x_hi: float,
    y_lo: float,
    y_hi: float,
    z_lo: float,
    z_hi: float,
    domain: Rect,
    max_steps: int,
) -> bool:
    """8-corner safety oracle for the unicycle."""
    corners = [
        (x_lo, y_lo, z_lo), (x_lo, y_lo, z_hi),
        (x_lo, y_hi, z_lo), (x_lo, y_hi, z_hi),
        (x_hi, y_lo, z_lo), (x_hi, y_lo, z_hi),
        (x_hi, y_hi, z_lo), (x_hi, y_hi, z_hi),
    ]

    for (cx, cy, cz) in corners:
        x = np.array([cx, cy, cz], dtype=float)
        reached_goal = False
        for _ in range(max_steps):
            # obstacle -> immediate fail for the whole cell
            if (x[0] - Y_OBS[0]) ** 2 + (x[1] - Y_OBS[1]) ** 2 <= R_OBS ** 2:
                return False
            # goal -> this corner succeeded; continue checking the rest
            if (x[0] - Y_GOAL[0]) ** 2 + (x[1] - Y_GOAL[1]) ** 2 <= R_GOAL ** 2:
                reached_goal = True
                break
            # leaving the domain before reaching goal is a fail
            if not (domain.xmin <= x[0] <= domain.xmax and domain.ymin <= x[1] <= domain.ymax):
                break
            x = unicycle_dynamics(x)
        if not reached_goal:
            return False
    return True


def build_gt_safe_set(
    nx_gt: int = 90,
    ny_gt: int = 90,
    nz_gt: int = 90,
    max_steps: int = 10,
    *,
    progress_every: int = 10,
) -> Set[Tuple[int, int, int]]:
    """Build the unicycle ground-truth safe set once from the main-code dynamics."""
    domain = _rect_domain()
    dx_gt = (domain.xmax - domain.xmin) / nx_gt
    dy_gt = (domain.ymax - domain.ymin) / ny_gt
    dz_gt = (domain.zmax - domain.zmin) / nz_gt

    gt_safe: Set[Tuple[int, int, int]] = set()
    total_cells = nx_gt * ny_gt * nz_gt
    done = 0

    print(
        f"[GT] Building unicycle GT safe set on {nx_gt}x{ny_gt}x{nz_gt} grid "
        f"(max_steps={max_steps})...",
        flush=True,
    )
    for i in range(nx_gt):
        if progress_every and i % progress_every == 0:
            print(f"  [GT] column {i}/{nx_gt}", flush=True)
        x_lo = domain.xmin + i * dx_gt
        x_hi = domain.xmin + (i + 1) * dx_gt
        for j in range(ny_gt):
            y_lo = domain.ymin + j * dy_gt
            y_hi = domain.ymin + (j + 1) * dy_gt
            for k in range(nz_gt):
                z_lo = domain.zmin + k * dz_gt
                z_hi = domain.zmin + (k + 1) * dz_gt
                if corners_are_safe(x_lo, x_hi, y_lo, y_hi, z_lo, z_hi, domain, max_steps=max_steps):
                    gt_safe.add((i, j, k))
                done += 1

    print(f"[GT] safe cells: {len(gt_safe)} / {total_cells}", flush=True)
    return gt_safe


def save_gt_cache(
    path: str | Path,
    gt_safe: Set[Tuple[int, int, int]],
    *,
    nx_gt: int,
    ny_gt: int,
    nz_gt: int,
    max_steps: int,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_version": GT_CACHE_VERSION,
        "nx_gt": nx_gt,
        "ny_gt": ny_gt,
        "nz_gt": nz_gt,
        "max_steps": max_steps,
        "domain": _rect_domain(),
        "gt_safe": gt_safe,
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)
    print(f"[GT] wrote cache to {path}", flush=True)
    return path


def load_gt_cache(path: str | Path) -> dict:
    path = Path(path)
    with path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"GT cache {path} is not a dict payload.")
    required = {"cache_version", "nx_gt", "ny_gt", "nz_gt", "max_steps", "domain", "gt_safe"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"GT cache {path} missing keys: {sorted(missing)}")
    if int(payload["cache_version"]) != GT_CACHE_VERSION:
        raise ValueError(
            f"GT cache version mismatch: {payload['cache_version']} != {GT_CACHE_VERSION}"
        )
    return payload


def load_or_build_gt_cache(
    path: str | Path,
    *,
    nx_gt: int = 100,
    ny_gt: int = 100,
    nz_gt: int = 100,
    max_steps: int = 10,
    force_rebuild: bool = False,
) -> dict:
    path = Path(path)
    if path.exists() and not force_rebuild:
        payload = load_gt_cache(path)
        if (
            int(payload["nx_gt"]) == nx_gt and
            int(payload["ny_gt"]) == ny_gt and
            int(payload["nz_gt"]) == nz_gt and
            int(payload["max_steps"]) == max_steps
        ):
            print(f"[GT] loaded cache from {path}", flush=True)
            return payload
        print(
            f"[GT] cache {path} exists but grid/steps do not match requested "
            f"{nx_gt}x{ny_gt}x{nz_gt}, steps={max_steps}; rebuilding...",
            flush=True,
        )

    gt_safe = build_gt_safe_set(nx_gt=nx_gt, ny_gt=ny_gt, nz_gt=nz_gt, max_steps=max_steps)
    save_gt_cache(path, gt_safe, nx_gt=nx_gt, ny_gt=ny_gt, nz_gt=nz_gt, max_steps=max_steps)
    return load_gt_cache(path)
from typing import Iterable, Optional


def _cell_to_gt_index(cell, payload):
    """
    Convert an abstraction Rect into GT-grid index ranges.
    """
    domain = payload["domain"]

    nx = payload["nx_gt"]
    ny = payload["ny_gt"]
    nz = payload["nz_gt"]

    dx = (domain.xmax - domain.xmin) / nx
    dy = (domain.ymax - domain.ymin) / ny
    dz = (domain.zmax - domain.zmin) / nz

    ix0 = max(0, min(nx - 1, int(np.floor((cell.xmin - domain.xmin) / dx))))
    ix1 = max(0, min(nx - 1, int(np.floor((cell.xmax - domain.xmin) / dx))))

    iy0 = max(0, min(ny - 1, int(np.floor((cell.ymin - domain.ymin) / dy))))
    iy1 = max(0, min(ny - 1, int(np.floor((cell.ymax - domain.ymin) / dy))))

    iz0 = max(0, min(nz - 1, int(np.floor((cell.zmin - domain.zmin) / dz))))
    iz1 = max(0, min(nz - 1, int(np.floor((cell.zmax - domain.zmin) / dz))))

    return ix0, ix1, iy0, iy1, iz0, iz1


def cell_is_fully_gt_safe(cell, payload):
    """
    True iff every GT voxel intersecting this abstraction cell is safe.
    """
    safe = payload["gt_safe"]

    ix0, ix1, iy0, iy1, iz0, iz1 = _cell_to_gt_index(cell, payload)

    for i in range(ix0, ix1 + 1):
        for j in range(iy0, iy1 + 1):
            for k in range(iz0, iz1 + 1):
                if (i, j, k) not in safe:
                    return False

    return True


def cell_overlaps_gt_unsafe(cell, payload):
    """
    True iff any GT voxel intersecting this abstraction cell is unsafe.
    """
    safe = payload["gt_safe"]

    ix0, ix1, iy0, iy1, iz0, iz1 = _cell_to_gt_index(cell, payload)

    for i in range(ix0, ix1 + 1):
        for j in range(iy0, iy1 + 1):
            for k in range(iz0, iz1 + 1):
                if (i, j, k) not in safe:
                    return True

    return False


def path_is_fully_gt_safe(absys, uid_path, gt_cache_payload):
    payload = gt_cache_payload
    for uid in uid_path:
        if uid == absys.OUT_UID:
            return False

        cell = absys.part.leaves[uid].rect

        if not cell_is_fully_gt_safe(cell, payload):
            return False

    return True


def first_path_uid_overlapping_unsafe(absys, uid_path, gt_cache_payload):
    payload = gt_cache_payload
    for uid in uid_path:
        if uid == absys.OUT_UID:
            continue

        cell = absys.part.leaves[uid].rect

        if cell_overlaps_gt_unsafe(cell, payload):
            return uid

    return None
