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
INITIAL_DOMAIN_UB = np.array([-8.0, -8.0])

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