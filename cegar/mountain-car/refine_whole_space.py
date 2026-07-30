# refine_whole_space.py
import gc
import random
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from abstraction import Abstraction, Rect

from cegar_loop_mc import run_cegar


def rect_volume(r) -> float:
    if r.zmin == r.zmax:
        return float((r.xmax - r.xmin) * (r.ymax - r.ymin))
    return float((r.xmax - r.xmin) * (r.ymax - r.ymin) * (r.zmax - r.zmin))


def cell_volume(absys: Abstraction, uid: int) -> float:
    return rect_volume(absys.part.leaves[uid].rect)


def is_verifiable(absys: Abstraction, uid: int) -> bool:
    if uid == absys.OUT_UID:
        return False
    if uid not in absys.part.leaves:
        return False
    labs = absys.ap_labeler(absys.part.leaves[uid].rect)
    return "unsafe" not in labs


def compute_slp(absys: Abstraction) -> float:
    total = 0
    self_loops = 0
    for uid, by_a in absys.tr.succ.items():
        if uid == absys.OUT_UID:
            continue
        for a, succs in by_a.items():
            total += len(succs)
            if uid in succs:
                self_loops += 1
    slp = self_loops / total if total > 0 else 0.0
    print(f"  SLP: {slp:.3f}  (self_loops={self_loops}, total_transitions={total})")
    return slp


def compute_msu(absys: Abstraction) -> float:
    counts = []
    for uid, by_a in absys.tr.succ.items():
        if uid == absys.OUT_UID:
            continue
        all_succs = set().union(*by_a.values()) if by_a else set()
        counts.append(len(all_succs))
    msu = sum(counts) / len(counts) if counts else 0.0
    print(f"  mSu: {msu:.2f}")
    return msu


def _rect_within_domain(xmin, xmax, ymin, ymax, zmin, zmax, domain) -> bool:
    """
    True iff the given box is fully contained within `domain`'s bounds.
    Used as an early-exit guard before running any (expensive) safety
    simulation or volume/satisfaction computation on a cell -- cells that
    fall outside the domain (or straddle its boundary) are skipped rather
    than silently evaluated as if they were valid interior cells.
    """
    return (domain.xmin <= xmin and xmax <= domain.xmax and
            domain.ymin <= ymin and ymax <= domain.ymax and
            domain.zmin <= zmin and zmax <= domain.zmax)


def corners_are_safe(x_lo, x_hi, y_lo, y_hi, z_lo, z_hi, domain, max_steps=300):
    """
    Ground-truth safety check for a box [x_lo,x_hi] x [y_lo,y_hi] x
    [z_lo,z_hi] (theta), by propagating ALL 8 CORNERS of the box through
    unicycle_dynamics (previously only the single center point at a fixed
    theta=0 was checked -- this both replaces the center-point check with
    genuine corner propagation, AND replaces the fixed theta=0 slice with
    the box's actual theta range):
    """
    from main import unicycle_dynamics, Y_GOAL, Y_OBS, R_GOAL, R_OBS

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
            # Obstacle hit -> immediate veto for the WHOLE box, regardless
            # of the remaining corners.
            if (x[0] - Y_OBS[0])**2 + (x[1] - Y_OBS[1])**2 <= R_OBS**2:
                return False
            if (x[0] - Y_GOAL[0])**2 + (x[1] - Y_GOAL[1])**2 <= R_GOAL**2:
                reached_goal = True
                break
            if not (domain.xmin <= x[0] <= domain.xmax and
                    domain.ymin <= x[1] <= domain.ymax):
                break  # left domain without reaching goal -> this corner fails
            x = unicycle_dynamics(x)
        if not reached_goal:
            # This corner neither hit the obstacle nor reached the goal in
            # time (ran out of steps, or left the domain) -> box not safe.
            return False

    return True  # every corner reached the goal in time, none hit the obstacle


def build_gt_safe_set(domain, nx_gt=60, ny_gt=60, nz_gt=20, max_steps=300):

    print(f"\n[GT] Building ground truth safe set ({nx_gt}x{ny_gt}x{nz_gt}, "
          f"8-corner check)...", flush=True)
    dx_gt = (domain.xmax - domain.xmin) / nx_gt
    dy_gt = (domain.ymax - domain.ymin) / ny_gt
    dz_gt = (domain.zmax - domain.zmin) / nz_gt

    gt_safe = set()
    total_cells = nx_gt * ny_gt * nz_gt
    done = 0
    for i in range(nx_gt):
        if i % 10 == 0:
            print(f"  [ground truth] col {i}/{nx_gt}", flush=True)
        x_lo = domain.xmin + i * dx_gt
        x_hi = domain.xmin + (i + 1) * dx_gt
        for j in range(ny_gt):
            y_lo = domain.ymin + j * dy_gt
            y_hi = domain.ymin + (j + 1) * dy_gt
            for k in range(nz_gt):
                z_lo = domain.zmin + k * dz_gt
                z_hi = domain.zmin + (k + 1) * dz_gt
                if corners_are_safe(x_lo, x_hi, y_lo, y_hi, z_lo, z_hi,
                                     domain, max_steps=max_steps):
                    gt_safe.add((i, j, k))
                done += 1

    print(f"  GT safe cells: {len(gt_safe)} / {total_cells}", flush=True)
    return gt_safe


def compute_tpr(
    absys,
    verified_now,
    domain,
    gt_safe,
    nx_gt=60,
    ny_gt=60,
    nz_gt=20,
    initial_domain=None,
):
    if initial_domain is None:
        initial_domain = domain

    dx_gt = (domain.xmax - domain.xmin) / nx_gt
    dy_gt = (domain.ymax - domain.ymin) / ny_gt
    dz_gt = (domain.zmax - domain.zmin) / nz_gt

    def cell_is_truly_safe(r) -> bool:
        eps = 1e-9
        i_lo = max(0, int((r.xmin - domain.xmin) / dx_gt))
        i_hi = min(nx_gt - 1, int((r.xmax - domain.xmin) / dx_gt - eps))
        j_lo = max(0, int((r.ymin - domain.ymin) / dy_gt))
        j_hi = min(ny_gt - 1, int((r.ymax - domain.ymin) / dy_gt - eps))
        k_lo = max(0, int((r.zmin - domain.zmin) / dz_gt))
        k_hi = min(nz_gt - 1, int((r.zmax - domain.zmin) / dz_gt - eps))
        for ii in range(i_lo, i_hi + 1):
            for jj in range(j_lo, j_hi + 1):
                for kk in range(k_lo, k_hi + 1):
                    if (ii, jj, kk) not in gt_safe:
                        return False
        return True

    truly_safe  = 0
    true_pos    = 0
    total_cells = 0

    for uid, node in absys.part.leaves.items():
        if uid == absys.OUT_UID:
            continue
        r = node.rect
        # --- Same initial-domain skip as compute_recall ---
        if not _rect_within_domain(r.xmin, r.xmax, r.ymin, r.ymax,
                                    r.zmin, r.zmax, initial_domain):
            continue
        total_cells += 1
        if not cell_is_truly_safe(r):
            continue
        truly_safe += 1
        if uid in verified_now:
            true_pos += 1

    tpr = true_pos / truly_safe if truly_safe > 0 else 0.0

    print(f"\n── TPR (paper Eq. 29, current partition) ─────────────────────────")
    print(f"  Ground truth grid:           {nx_gt}x{ny_gt}x{nz_gt} = {nx_gt*ny_gt*nz_gt} cells")
    print(f"  GT safe cells (gt grid):     {len(gt_safe)}")
    print(f"  Current cells (total):       {total_cells}")
    print(f"  Current cells in GT-safe:    {truly_safe}  [denominator]")
    print(f"  Verified AND GT-safe:        {true_pos}  [numerator]")
    print(f"  TPR:                         {tpr:.4f}  ({tpr*100:.2f}%)")
    print(f"──────────────────────────────────────────────────────────────────")

    return tpr


def compute_recall(absys, verified_now, domain, gt_safe, nx_gt=60, ny_gt=60,
                    nz_gt=20, initial_domain=None):

    if initial_domain is None:
        initial_domain = domain

    dx_gt = (domain.xmax - domain.xmin) / nx_gt
    dy_gt = (domain.ymax - domain.ymin) / ny_gt
    dz_gt = (domain.zmax - domain.zmin) / nz_gt

    def cell_is_truly_safe(r) -> bool:
        """Same all-overlapping-GT-cells check as compute_tpr (now 3D),
        including the epsilon fix for boundary-aligned GT cells."""
        eps = 1e-9
        i_lo = max(0, int((r.xmin - domain.xmin) / dx_gt))
        i_hi = min(nx_gt - 1, int((r.xmax - domain.xmin) / dx_gt - eps))
        j_lo = max(0, int((r.ymin - domain.ymin) / dy_gt))
        j_hi = min(ny_gt - 1, int((r.ymax - domain.ymin) / dy_gt - eps))
        k_lo = max(0, int((r.zmin - domain.zmin) / dz_gt))
        k_hi = min(nz_gt - 1, int((r.zmax - domain.zmin) / dz_gt - eps))
        for ii in range(i_lo, i_hi + 1):
            for jj in range(j_lo, j_hi + 1):
                for kk in range(k_lo, k_hi + 1):
                    if (ii, jj, kk) not in gt_safe:
                        return False
        return True

    verified_vol = 0.0
    gt_safe_vol  = 0.0

    for uid, node in absys.part.leaves.items():
        if uid == absys.OUT_UID:
            continue
        r = node.rect
        # --- FIX: skip any cell that doesn't fall within initial_domain,
        # BEFORE doing any satisfaction check or volume calculation. ---
        if not _rect_within_domain(r.xmin, r.xmax, r.ymin, r.ymax,
                                    r.zmin, r.zmax, initial_domain):
            continue
        if not cell_is_truly_safe(r):
            continue
        vol = rect_volume(r)
        gt_safe_vol += vol
        if uid in verified_now:
            verified_vol += vol

    recall = verified_vol / gt_safe_vol if gt_safe_vol > 0 else 0.0

    print(f"\n── Recall (verified vol / GT-safe vol) ────────────────────────────")
    print(f"  Verified volume (GT-safe cells only): {verified_vol:.4f}")
    print(f"  GT-safe volume (current partition):   {gt_safe_vol:.4f}")
    print(f"  Recall:                               {recall:.4f}  ({recall*100:.2f}%)")
    print(f"──────────────────────────────────────────────────────────────────")

    return recall


@dataclass
class RegionClassification:
    verified: Set[int]
    refuted: Set[int]
    unknown: Set[int]


def compute_verified_set_via_fixpoint(absys: Abstraction, action: str = "step") -> Set[int]:
    """
    Computes the set of cells satisfying (!unsafe) U goal for ALL paths,
    via a single global backward fixpoint over the transition graph.
    """
    leaves = absys.part.leaves
    all_aps, labels_by_uid = absys.aps_and_labels()

    def is_safe(uid: int) -> bool:
        if uid == absys.OUT_UID:
            return False
        if uid not in leaves:
            return False
        return "unsafe" not in labels_by_uid.get(uid, set())

    def is_goal(uid: int) -> bool:
        return "goal" in labels_by_uid.get(uid, set())

    succs: Dict[int, Set[int]] = {}
    for uid in leaves.keys():
        if uid == absys.OUT_UID:
            continue
        by_a = absys.tr.succ.get(uid, {})
        merged: Set[int] = set()
        for a, s in by_a.items():
            merged |= s
        succs[uid] = merged

    predecessors: Dict[int, List[int]] = {}
    for s, ts in succs.items():
        for t in ts:
            predecessors.setdefault(t, []).append(s)

    remaining: Dict[int, int] = {s: len(ts) for s, ts in succs.items()}

    W: Set[int] = set()
    queue: deque = deque()

    for uid in leaves.keys():
        if uid == absys.OUT_UID:
            continue
        if is_safe(uid) and is_goal(uid):
            W.add(uid)
            queue.append(uid)

    while queue:
        t = queue.popleft()
        for s in predecessors.get(t, []):
            if s in W or not is_safe(s):
                continue
            remaining[s] -= 1
            if remaining[s] <= 0:
                W.add(s)
                queue.append(s)

    return W


def classify_all_leaves_once(
    absys: Abstraction,
    phi: str,
    action: str = "step",
) -> RegionClassification:
    print("\n[CLASSIFY] computing global fixpoint (replaces per-cell Spot calls)...",
          flush=True)

    refuted: Set[int] = set()
    leaves = [u for u in absys.part.leaves.keys() if u != absys.OUT_UID]
    for uid in leaves:
        if not is_verifiable(absys, uid):
            refuted.add(uid)

    verified = compute_verified_set_via_fixpoint(absys, action=action)
    unknown  = set(leaves) - verified - refuted

    print(f"  [classify] done. verified={len(verified)} "
          f"refuted={len(refuted)} unknown={len(unknown)}", flush=True)

    return RegionClassification(verified, refuted, unknown)


def compute_metrics(absys, verified_now, refuted_now, unknown_now):
    current_leaves = [u for u in absys.part.leaves.keys() if u != absys.OUT_UID]

    verified_vol = sum(rect_volume(absys.part.leaves[u].rect) for u in verified_now)
    refuted_vol  = sum(rect_volume(absys.part.leaves[u].rect) for u in refuted_now)
    unknown_vol  = sum(rect_volume(absys.part.leaves[u].rect) for u in unknown_now)
    total_vol    = verified_vol + refuted_vol + unknown_vol

    resolvable_cells = len(verified_now) + len(unknown_now)
    resolvable_vol   = verified_vol + unknown_vol

    recall_cells  = len(verified_now) / resolvable_cells if resolvable_cells > 0 else 0.0
    FNR_cells     = len(unknown_now)  / resolvable_cells if resolvable_cells > 0 else 0.0
    recall_vol    = verified_vol / resolvable_vol if resolvable_vol > 0 else 0.0
    FNR_vol       = unknown_vol  / resolvable_vol if resolvable_vol > 0 else 0.0
    SR_total_vol  = verified_vol / total_vol if total_vol > 0 else 0.0
    FNR_total_vol = unknown_vol  / total_vol if total_vol > 0 else 0.0

    print("\n── Verification Metrics ──────────────────────────────────────────")
    print(f"  Total cells (excl. OUT):     {len(current_leaves)}")
    print(f"  Verified cells:              {len(verified_now)}")
    print(f"  Refuted  cells:              {len(refuted_now)}")
    print(f"  Unknown  cells:              {len(unknown_now)}")
    print()
    print(f"  Verified volume:             {verified_vol:.4f}")
    print(f"  Refuted  volume:             {refuted_vol:.4f}")
    print(f"  Unknown  volume:             {unknown_vol:.4f}")
    print(f"  Total    volume:             {total_vol:.4f}")
    print()
    print(f"  Recall / SR  (cell count):   {recall_cells:.4f}  ({recall_cells*100:.2f}%)")
    print(f"  FNR          (cell count):   {FNR_cells:.4f}  ({FNR_cells*100:.2f}%)")
    print()
    print(f"  Recall / SR  (volume):       {recall_vol:.4f}  ({recall_vol*100:.2f}%)")
    print(f"  FNR          (volume):       {FNR_vol:.4f}  ({FNR_vol*100:.2f}%)")
    print()
    print(f"  SR  / total domain volume:   {SR_total_vol:.4f}  ({SR_total_vol*100:.2f}%)")
    print(f"  FNR / total domain volume:   {FNR_total_vol:.4f}  ({FNR_total_vol*100:.2f}%)")
    print("──────────────────────────────────────────────────────────────────")

    return {
        "recall_cells":   recall_cells,
        "FNR_cells":      FNR_cells,
        "recall_vol":     recall_vol,
        "FNR_vol":        FNR_vol,
        "SR_total_vol":   SR_total_vol,
        "FNR_total_vol":  FNR_total_vol,
        "verified_cells": len(verified_now),
        "refuted_cells":  len(refuted_now),
        "unknown_cells":  len(unknown_now),
        "verified_vol":   verified_vol,
        "refuted_vol":    refuted_vol,
        "unknown_vol":    unknown_vol,
        "total_vol":      total_vol,
    }


def refine_from_unknowns_single_pass(
    absys: Abstraction,
    phi: str,
    unknown_uids: Set[int],
    max_iters_per_cell: int = 150,
    min_cell_width: float = 0.001,
    min_cell_height: float = 0.001,
    max_refine_depth: int = 40,
    min_cell_theta: float = None,
    split_mode: str = "auto",
    gc_every: int = 100,
    ordering: str = "largest",
    rand_seed: int = 0,
) -> int:
    """
    ordering: "largest"  — largest cells first (original default)
              "smallest" — smallest cells first (inverted)
              "random"   — shuffled with rand_seed
    """
    candidates = [u for u in unknown_uids if u in absys.part.leaves]

    if ordering == "largest":
        ordered = sorted(candidates, key=lambda u: (-cell_volume(absys, u), u))
    elif ordering == "smallest":
        largest_order = sorted(candidates, key=lambda u: (-cell_volume(absys, u), u))
        ordered = list(reversed(largest_order))
    elif ordering == "random":
        ordered = list(candidates)
        random.seed(rand_seed)
        random.shuffle(ordered)
    else:
        raise ValueError(f"Unknown ordering: {ordering!r}. Use 'largest', 'smallest', or 'random'.")

    total = len(ordered)
    total_refinements   = 0
    total_iters         = 0
    total_ignored       = 0
    n_verified_by_cegar = 0
    n_no_splits         = 0

    print(f"\n[REFINE] Single-pass over {total} cells (ordering={ordering!r}"
          + (f", seed={rand_seed}" if ordering == "random" else "")
          + f", split_mode={split_mode!r}"
          + f", min_cell_theta={min_cell_theta}"
          + ")...", flush=True)
    for i, uid in enumerate(ordered):
        if i % 200 == 0:
            print(f"  [refine] {i}/{total} leaves={len(absys.part.leaves)} "
                  f"refinements_so_far={total_refinements}", flush=True)

        if uid not in absys.part.leaves:
            continue

        res = run_cegar(
            absys=absys,
            init_uids={uid},
            phi=phi,
            max_iters=max_iters_per_cell,
            merge_actions=True,
            min_cell_width=min_cell_width,
            min_cell_height=min_cell_height,
            max_refine_depth=max_refine_depth,
            min_cell_theta=min_cell_theta,
            split_mode=split_mode,
            verbose=False,
        )
        total_refinements  += getattr(res, "refinements", 0)
        total_iters        += getattr(res, "iterations", 0)
        total_ignored       += getattr(res, "ignored_counterexamples", 0)
        if res.verified:
            n_verified_by_cegar += 1
        elif getattr(res, "refinements", 0) == 0:
            n_no_splits += 1

        if i % gc_every == 0:
            gc.collect()

    print(f"[REFINE] pass done.", flush=True)
    print(f"  Total cell splits:          {total_refinements}")
    print(f"  Total CEGAR iters:          {total_iters}")
    print(f"  Total ignored CEs:          {total_ignored}")
    print(f"  Cells verified by CEGAR:    {n_verified_by_cegar}")
    print(f"  Cells with 0 splits:        {n_no_splits}")
    print(f"  Leaves:                     {len(absys.part.leaves)}", flush=True)
    return total_refinements


def refine_one_round(
    absys: Abstraction,
    phi: str,
    initial_unknown: Set[int],
    max_iters_per_cell: int = 150,
    min_cell_width: float = 0.001,
    min_cell_height: float = 0.001,
    max_refine_depth: int = 40,
    min_cell_theta: float = 0.001,
    split_mode: str = "auto",
    gc_every: int = 100,
    ordering: str = "largest",
    rand_seed: int = 0,
) -> RegionClassification:
    """Single round of refinement"""
    print(f"\n{'='*60}", flush=True)
    print(f"[ROUND 1/1] unknown={len(initial_unknown)} "
          f"leaves={len(absys.part.leaves)} ordering={ordering!r} "
          f"split_mode={split_mode!r}", flush=True)
    print(f"{'='*60}", flush=True)

    if initial_unknown:
        refine_from_unknowns_single_pass(
            absys=absys,
            phi=phi,
            unknown_uids=initial_unknown,
            max_iters_per_cell=max_iters_per_cell,
            min_cell_width=min_cell_width,
            min_cell_height=min_cell_height,
            max_refine_depth=max_refine_depth,
            min_cell_theta=min_cell_theta,
            split_mode=split_mode,
            gc_every=gc_every,
            ordering=ordering,
            rand_seed=rand_seed,
        )
    else:
        print("[ROUND 1/1] No unknowns to refine.", flush=True)

    print(f"\n[FINAL CLASSIFY]", flush=True)
    return classify_all_leaves_once(absys, phi, action="step")


def visualize_classification(
    absys: Abstraction,
    verified: set,
    refuted: set,
    title: str = "State Space Classification",
    save_path: str | None = None,
    show_grid: bool = True,
    grid_linewidth: float = 0.5,
    show_goal_border: bool = True,
    goal_border_color: str = "blue",
    goal_border_linewidth: float = 2.5,
):

    COLOR_CELL = "#d3d3d3"   # light gray

    fig, ax = plt.subplots(figsize=(10, 10))
    all_aps, labels_by_uid = absys.aps_and_labels()

    # ── Layer 0: Gray fill for every unique xy footprint ─────────────────────
    seen_xy: set = set()
    for uid, node in absys.part.leaves.items():
        if uid == absys.OUT_UID:
            continue
        r   = node.rect
        key = (round(r.xmin, 6), round(r.xmax, 6),
               round(r.ymin, 6), round(r.ymax, 6))
        if key in seen_xy:
            continue
        seen_xy.add(key)
        ax.add_patch(patches.Rectangle(
            (r.xmin, r.ymin), r.xmax - r.xmin, r.ymax - r.ymin,
            linewidth=0.0, edgecolor=None,
            facecolor=COLOR_CELL, alpha=1.0, zorder=1,
        ))

    # ── Layer 1: Grid lines ──────────────────────────────────────────────────
    if show_grid:
        drawn: set = set()
        for uid, node in absys.part.leaves.items():
            r   = node.rect
            key = (round(r.xmin, 6), round(r.xmax, 6),
                   round(r.ymin, 6), round(r.ymax, 6))
            if key in drawn:
                continue
            drawn.add(key)
            ax.add_patch(patches.Rectangle(
                (r.xmin, r.ymin), r.xmax - r.xmin, r.ymax - r.ymin,
                linewidth=grid_linewidth, edgecolor="black",
                facecolor="none", alpha=0.6, zorder=2,
            ))

    # ── Layer 2: Goal region border (blue outline) ───────────────────────────
    if show_goal_border:
        drawn_goal: set = set()
        for uid, node in absys.part.leaves.items():
            if uid == absys.OUT_UID:
                continue
            if "goal" not in labels_by_uid.get(uid, set()):
                continue
            r   = node.rect
            key = (round(r.xmin, 6), round(r.xmax, 6),
                   round(r.ymin, 6), round(r.ymax, 6))
            if key in drawn_goal:
                continue
            drawn_goal.add(key)
            ax.add_patch(patches.Rectangle(
                (r.xmin, r.ymin), r.xmax - r.xmin, r.ymax - r.ymin,
                linewidth=goal_border_linewidth, edgecolor=goal_border_color,
                facecolor="none", alpha=1.0, zorder=3,
            ))

    ax.set_xlim(0, 50)
    ax.set_ylim(0, 50)
    ax.set_xlabel("y1")
    ax.set_ylabel("y2")
    ax.set_title(title)
    ax.grid(False)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[VIS] saved classification to {save_path}")

    plt.show()

