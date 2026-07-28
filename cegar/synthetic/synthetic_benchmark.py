"""
synthetic_benchmark.py

Benchmark interface for the synthetic affine system.

Part 1:
    - configuration
    - dynamics wrapper
    - AP labeler
    - abstraction builder
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from abstraction import (
    Rect,
    RectPartition,
    Abstraction,
    AffineDynamics,
)

import synthetic_system as ss

###############################################################################
# Benchmark constants
###############################################################################

DOMAIN_LB = np.array([-10.0, -10.0])
DOMAIN_UB = np.array([10.0, 10.0])

INITIAL_DOMAIN_LB = np.array([-10.0, -10.0])
INITIAL_DOMAIN_UB = np.array([10.0, 10.0])

DOMAIN = Rect(
    DOMAIN_LB[0],
    DOMAIN_UB[0],
    DOMAIN_LB[1],
    DOMAIN_UB[1],
)

INITIAL_DOMAIN = Rect(
    INITIAL_DOMAIN_LB[0],
    INITIAL_DOMAIN_UB[0],
    INITIAL_DOMAIN_LB[1],
    INITIAL_DOMAIN_UB[1],
)

GOAL_CENTER = np.array([5.0, 5.0])
GOAL_RADIUS = 2.0

DEFAULT_NX = 40
DEFAULT_NY = 40

FORMULA = "(!unsafe) U goal"

###############################################################################
# Dynamics wrapper
###############################################################################

class SyntheticDynamics(AffineDynamics):
    """
    Generic AffineDynamics wrapper around synthetic_system.py.
    """

    def __init__(self):
        super().__init__(
            A=ss.A_GLOBAL,
            xstar=GOAL_CENTER,
        )
        self.goal_radius = GOAL_RADIUS

    def dynamics(self, x):
        return ss.dynamics(
            np.asarray(x, dtype=float),
            GOAL_CENTER,
        )

    def image_bbox(self, rect):
        """
        Exact reachable AABB obtained by propagating the four corners.
        """

        corners = np.array([
            [rect.xmin, rect.ymin],
            [rect.xmin, rect.ymax],
            [rect.xmax, rect.ymin],
            [rect.xmax, rect.ymax],
        ])

        imgs = np.array([self.dynamics(c) for c in corners])

        xmin = float(imgs[:, 0].min())
        xmax = float(imgs[:, 0].max())

        ymin = float(imgs[:, 1].min())
        ymax = float(imgs[:, 1].max())

        return [
            Rect(
                xmin,
                xmax,
                ymin,
                ymax,
            )
        ]


###############################################################################
# Atomic proposition labeling
###############################################################################

def ap_labeler(rect):
    """
    OUT state:
        rect=None

    A cell is a goal iff ALL corners lie inside the goal ball.

    Otherwise it is simply safe.
    """

    if rect is None:
        return {"unsafe"}

    corners = np.array([
        [rect.xmin, rect.ymin],
        [rect.xmin, rect.ymax],
        [rect.xmax, rect.ymin],
        [rect.xmax, rect.ymax],
    ])

    dist = np.linalg.norm(
        corners - GOAL_CENTER,
        axis=1,
    )

    if np.all(dist <= GOAL_RADIUS):
        return {
            "goal",
            "safe",
        }

    return {
        "safe",
    }


###############################################################################
# Abstraction construction
###############################################################################

def build_abstraction(
    nx=DEFAULT_NX,
    ny=DEFAULT_NY,
):
    """
    Build the generic CEGAR abstraction for the synthetic benchmark.
    """

    partition = RectPartition.uniform_grid(
        DOMAIN,
        nx,
        ny,
    )

    dyn = {
        "step": SyntheticDynamics(),
    }

    absys = Abstraction(
        partition,
        dyn,
        ap_labeler,
    )

    absys.rebuild_all_transitions()

    return absys, DOMAIN
###############################################################################
# Volume helpers
###############################################################################

def rect_volume(r: Rect) -> float:
    return float(
        (r.xmax - r.xmin) *
        (r.ymax - r.ymin)
    )


def cell_volume(absys: Abstraction, uid: int) -> float:
    return rect_volume(absys.part.leaves[uid].rect)


###############################################################################
# Ground-truth safety
###############################################################################

def state_reaches_goal(
    x0,
    max_steps=200,
):
    """
    Ground-truth simulation.

    Because the synthetic system is deterministic and globally stable,
    every point is simulated until

        • enters goal
        • leaves domain
        • timeout
    """

    x = np.asarray(x0, dtype=float)

    for _ in range(max_steps):

        if np.linalg.norm(x - GOAL_CENTER) <= GOAL_RADIUS:
            return True

        if (
            x[0] < DOMAIN.xmin or
            x[0] > DOMAIN.xmax or
            x[1] < DOMAIN.ymin or
            x[1] > DOMAIN.ymax
        ):
            return False

        x = ss.dynamics(x, GOAL_CENTER)

    return False


def cell_is_safe(
    rect: Rect,
    max_steps=200,
):
    """
    Conservative ground-truth:

    all four corners must reach the goal.
    """

    corners = [
        np.array([rect.xmin, rect.ymin]),
        np.array([rect.xmin, rect.ymax]),
        np.array([rect.xmax, rect.ymin]),
        np.array([rect.xmax, rect.ymax]),
    ]

    for c in corners:
        if not state_reaches_goal(
            c,
            max_steps=max_steps,
        ):
            return False

    return True


def build_gt_safe_set(
    nx_gt=80,
    ny_gt=80,
    max_steps=200,
):
    """
    Returns a set of GT-safe grid cells.
    """

    print(
        f"\n[GT] Building {nx_gt}x{ny_gt} ground truth..."
    )

    dx = (DOMAIN.xmax - DOMAIN.xmin) / nx_gt
    dy = (DOMAIN.ymax - DOMAIN.ymin) / ny_gt

    gt_safe = set()

    for i in range(nx_gt):

        if i % 10 == 0:
            print(
                f"  row {i}/{nx_gt}",
                flush=True,
            )

        for j in range(ny_gt):

            r = Rect(
                DOMAIN.xmin + i * dx,
                DOMAIN.xmin + (i + 1) * dx,
                DOMAIN.ymin + j * dy,
                DOMAIN.ymin + (j + 1) * dy,
            )

            if cell_is_safe(
                r,
                max_steps=max_steps,
            ):
                gt_safe.add((i, j))

    print(
        f"GT safe cells: {len(gt_safe)}"
    )

    return gt_safe


###############################################################################
# Metrics
###############################################################################

def compute_tpr(
    absys,
    verified,
    gt_safe,
    nx_gt=80,
    ny_gt=80,
):
    """
    True-positive rate.
    """

    dx = (DOMAIN.xmax - DOMAIN.xmin) / nx_gt
    dy = (DOMAIN.ymax - DOMAIN.ymin) / ny_gt

    def covered(rect):

        eps = 1e-9

        ilo = max(
            0,
            int((rect.xmin - DOMAIN.xmin) / dx),
        )

        ihi = min(
            nx_gt - 1,
            int((rect.xmax - DOMAIN.xmin) / dx - eps),
        )

        jlo = max(
            0,
            int((rect.ymin - DOMAIN.ymin) / dy),
        )

        jhi = min(
            ny_gt - 1,
            int((rect.ymax - DOMAIN.ymin) / dy - eps),
        )

        for i in range(ilo, ihi + 1):
            for j in range(jlo, jhi + 1):
                if (i, j) not in gt_safe:
                    return False

        return True

    tp = 0
    safe = 0

    for uid, node in absys.part.leaves.items():

        if covered(node.rect):

            safe += 1

            if uid in verified:
                tp += 1

    tpr = tp / safe if safe else 0.0

    print(f"TPR = {tpr:.4f}")

    return tpr


def compute_recall(
    absys,
    verified,
    gt_safe,
    nx_gt=80,
    ny_gt=80,
):
    """
    Volume recall.
    """

    dx = (DOMAIN.xmax - DOMAIN.xmin) / nx_gt
    dy = (DOMAIN.ymax - DOMAIN.ymin) / ny_gt

    def covered(rect):

        eps = 1e-9

        ilo = max(
            0,
            int((rect.xmin - DOMAIN.xmin) / dx),
        )

        ihi = min(
            nx_gt - 1,
            int((rect.xmax - DOMAIN.xmin) / dx - eps),
        )

        jlo = max(
            0,
            int((rect.ymin - DOMAIN.ymin) / dy),
        )

        jhi = min(
            ny_gt - 1,
            int((rect.ymax - DOMAIN.ymin) / dy - eps),
        )

        for i in range(ilo, ihi + 1):
            for j in range(jlo, jhi + 1):
                if (i, j) not in gt_safe:
                    return False

        return True

    verified_vol = 0.0
    safe_vol = 0.0

    for uid, node in absys.part.leaves.items():

        if not covered(node.rect):
            continue

        vol = rect_volume(node.rect)

        safe_vol += vol

        if uid in verified:
            verified_vol += vol

    recall = verified_vol / safe_vol if safe_vol else 0.0

    print(f"Recall = {recall:.4f}")

    return recall


def _synthetic_v3_fixed_index_range(edges, lo, hi):
    """Exact index convention used by synthetic-v3.check_ground_truth_fast."""
    n = len(edges) - 1
    if hi <= edges[0] or lo >= edges[-1]:
        return None
    i_lo = int(np.searchsorted(edges, lo, side="right") - 1)
    i_hi = int(np.searchsorted(edges, hi, side="left") - 1)
    i_lo = max(0, min(n - 1, i_lo))
    i_hi = max(0, min(n - 1, i_hi))
    if i_hi < i_lo:
        return None
    return i_lo, i_hi


def classify_partition_with_synthetic_v3_ground_truth(
    absys,
    gt_reach_regions,
    *,
    domain=DOMAIN,
):
    """Label hierarchical leaves using synthetic-v3's fixed-grid rule.

    A leaf is ``goal`` only when every fixed ground-truth cell with
    positive-area overlap is labeled ``goal``.  This is the hierarchical
    partition equivalent of synthetic-v3's ``check_ground_truth_fast``.
    """
    if not gt_reach_regions:
        raise ValueError("gt_reach_regions is empty")
    if not all(
        isinstance(key, tuple) and len(key) == 2
        for key in gt_reach_regions
    ):
        raise ValueError(
            "Expected gt_reach_regions keys of the form (i, j)."
        )

    nx_gt = max(int(key[0]) for key in gt_reach_regions) + 1
    ny_gt = max(int(key[1]) for key in gt_reach_regions) + 1
    expected_keys = nx_gt * ny_gt
    if len(gt_reach_regions) != expected_keys:
        raise ValueError(
            "Ground-truth grid is incomplete: "
            f"found {len(gt_reach_regions)} cells, expected {expected_keys}."
        )

    x_edges = np.linspace(domain.xmin, domain.xmax, nx_gt + 1)
    y_edges = np.linspace(domain.ymin, domain.ymax, ny_gt + 1)
    goal_cells = {
        (int(key[0]), int(key[1]))
        for key, label in gt_reach_regions.items()
        if label == "goal"
    }

    classification = {}
    for uid, node in absys.part.leaves.items():
        rect = node.rect
        area = rect_volume(rect)
        if (
            rect.xmin < domain.xmin
            or rect.xmax > domain.xmax
            or rect.ymin < domain.ymin
            or rect.ymax > domain.ymax
        ):
            classification[uid] = ("fail", area)
            continue

        x_range = _synthetic_v3_fixed_index_range(
            x_edges,
            rect.xmin,
            rect.xmax,
        )
        y_range = _synthetic_v3_fixed_index_range(
            y_edges,
            rect.ymin,
            rect.ymax,
        )
        if x_range is None or y_range is None:
            classification[uid] = ("fail", area)
            continue

        all_goal = all(
            (i, j) in goal_cells
            for i in range(x_range[0], x_range[1] + 1)
            for j in range(y_range[0], y_range[1] + 1)
        )
        classification[uid] = (
            ("goal", area) if all_goal else ("fail", area)
        )

    return classification


def compute_synthetic_v3_recall(
    absys,
    verified,
    gt_reach_regions,
    *,
    initial_domain=DOMAIN,
    domain=DOMAIN,
):
    """Compute the same volume recall reported by synthetic-v3.

    The denominator is the volume of initial abstraction leaves classified
    ``goal`` by the fixed ground truth.  The numerator is the portion of that
    same volume contained in model-checked/verified initial leaves.
    """
    classification = classify_partition_with_synthetic_v3_ground_truth(
        absys,
        gt_reach_regions,
        domain=domain,
    )

    def is_initial(rect):
        # Match synthetic-v3's side="right" overlap convention.
        return (
            rect.xmax > initial_domain.xmin
            and rect.xmin <= initial_domain.xmax
            and rect.ymax > initial_domain.ymin
            and rect.ymin <= initial_domain.ymax
        )

    initial_uids = {
        uid
        for uid, node in absys.part.leaves.items()
        if is_initial(node.rect)
    }
    gt_safe_uids = {
        uid
        for uid, (label, _) in classification.items()
        if label == "goal" and uid in initial_uids
    }
    verified_gt_safe_uids = gt_safe_uids & set(verified)

    gt_safe_volume = sum(
        classification[uid][1]
        for uid in gt_safe_uids
    )
    verified_gt_safe_volume = sum(
        classification[uid][1]
        for uid in verified_gt_safe_uids
    )
    recall = (
        verified_gt_safe_volume / gt_safe_volume
        if gt_safe_volume
        else float("nan")
    )

    labels = {}
    for label in gt_reach_regions.values():
        labels[label] = labels.get(label, 0) + 1

    print(
        "Recall "
        f"(synthetic-v3) = {recall:.4f} "
        f"[{verified_gt_safe_volume:.12g}/{gt_safe_volume:.12g}]"
    )
    return {
        "recall": recall,
        "ground_truth_safe_volume": gt_safe_volume,
        "verified_ground_truth_safe_volume": verified_gt_safe_volume,
        "ground_truth_safe_partition_cells": len(gt_safe_uids),
        "verified_ground_truth_safe_partition_cells": len(
            verified_gt_safe_uids
        ),
        "initial_partition_cells": len(initial_uids),
        "gt_nx": max(int(key[0]) for key in gt_reach_regions) + 1,
        "gt_ny": max(int(key[1]) for key in gt_reach_regions) + 1,
        "gt_fixed_grid_label_counts": labels,
    }
###############################################################################
# Visualization
###############################################################################

def visualize_classification(
    absys,
    verified,
    refuted=None,
    title="Synthetic Benchmark",
    save_path=None,
    show_grid=True,
    grid_linewidth=0.5,
):
    """
    Visualize the current abstraction.

    Gray      : all abstract cells
    Green     : verified cells
    Blue      : goal region
    """

    if refuted is None:
        refuted = set()

    fig, ax = plt.subplots(figsize=(8, 8))

    ###########################################################################
    # Draw every cell
    ###########################################################################

    for uid, node in absys.part.leaves.items():

        r = node.rect

        color = "#d3d3d3"

        if uid in verified:
            color = "#66bb6a"

        elif uid in refuted:
            color = "#ef5350"

        ax.add_patch(
            patches.Rectangle(
                (r.xmin, r.ymin),
                r.xmax - r.xmin,
                r.ymax - r.ymin,
                facecolor=color,
                edgecolor="black" if show_grid else color,
                linewidth=grid_linewidth if show_grid else 0,
                alpha=0.85,
            )
        )

    ###########################################################################
    # Goal region
    ###########################################################################

    goal = plt.Circle(
        GOAL_CENTER,
        GOAL_RADIUS,
        fill=False,
        edgecolor="blue",
        linewidth=2.5,
    )

    ax.add_patch(goal)

    ###########################################################################

    ax.set_xlim(DOMAIN.xmin, DOMAIN.xmax)
    ax.set_ylim(DOMAIN.ymin, DOMAIN.ymax)

    ax.set_aspect("equal")

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")

    ax.set_title(title)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )
        print(f"[VIS] Saved figure to {save_path}")

    plt.show()


###############################################################################
# Convenience exports
###############################################################################

domain = DOMAIN
initial_domain = INITIAL_DOMAIN
phi = FORMULA


###############################################################################
# Benchmark API
###############################################################################

def ground_truth_safe_set(
    nx_gt=80,
    ny_gt=80,
    max_steps=200,
):
    return build_gt_safe_set(
        nx_gt=nx_gt,
        ny_gt=ny_gt,
        max_steps=max_steps,
    )

def refine_cell(absys, uid):

    r = absys.part.leaves[uid].rect

    xm = 0.5 * (r.xmin + r.xmax)
    ym = 0.5 * (r.ymin + r.ymax)

    absys.split_and_update(
        xm,
        ym,
        uid,
    )
def cell_is_mixed(
    rect,
    gt_safe,
    nx_gt=80,
    ny_gt=80,
):
    """
    True iff this abstract cell contains both GT-safe and GT-unsafe
    fine cells.
    """

    dx = (DOMAIN.xmax - DOMAIN.xmin) / nx_gt
    dy = (DOMAIN.ymax - DOMAIN.ymin) / ny_gt

    eps = 1e-9

    ilo = max(
        0,
        int((rect.xmin - DOMAIN.xmin) / dx),
    )

    ihi = min(
        nx_gt - 1,
        int((rect.xmax - DOMAIN.xmin) / dx - eps),
    )

    jlo = max(
        0,
        int((rect.ymin - DOMAIN.ymin) / dy),
    )

    jhi = min(
        ny_gt - 1,
        int((rect.ymax - DOMAIN.ymin) / dy - eps),
    )

    seen_safe = False
    seen_unsafe = False

    for i in range(ilo, ihi + 1):
        for j in range(jlo, jhi + 1):

            if (i, j) in gt_safe:
                seen_safe = True
            else:
                seen_unsafe = True

            if seen_safe and seen_unsafe:
                return True

    return False


###############################################################################
# Simple benchmark self-test
###############################################################################

if __name__ == "__main__":

    absys, domain = build_abstraction()

    print("Number of leaves:", len(absys.part.leaves))

    gt = build_gt_safe_set()

    print("GT safe cells:", len(gt))

    visualize_classification(
        absys,
        verified=set(),
        title="Synthetic Benchmark",
    )
