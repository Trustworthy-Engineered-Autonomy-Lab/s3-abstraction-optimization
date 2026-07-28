"""Mountain Car benchmark helpers shared by the production CEGAR runner."""
from __future__ import annotations

import numpy as np

from abstraction import Rect
import main_mountain_car_model_checking as mc
from refine_whole_space import compute_verified_set_via_fixpoint


DOMAIN = Rect(
    xmin=mc.X_MIN,
    xmax=mc.X_MAX,
    ymin=mc.Y_MIN,
    ymax=mc.Y_MAX,
)
INITIAL_DOMAIN = DOMAIN
FORMULA = "F goal"


def rect_area(rect: Rect) -> float:
    return float(
        (rect.xmax - rect.xmin)
        * (rect.ymax - rect.ymin)
    )


def _fixed_index_range(edges, lo, hi):
    """Exact convention from mountain-car-v3.check_ground_truth_fast."""
    count = len(edges) - 1
    if hi <= edges[0] or lo >= edges[-1]:
        return None
    first = int(np.searchsorted(edges, lo, side="right") - 1)
    last = int(np.searchsorted(edges, hi, side="left") - 1)
    first = max(0, min(count - 1, first))
    last = max(0, min(count - 1, last))
    if last < first:
        return None
    return first, last


def classify_partition_with_v3_ground_truth(
    absys,
    gt_reach_regions,
    *,
    domain: Rect = DOMAIN,
):
    """Classify hierarchical leaves using mountain-car-v3's fixed grid."""
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
    if len(gt_reach_regions) != nx_gt * ny_gt:
        raise ValueError(
            "Ground-truth grid is incomplete: "
            f"found {len(gt_reach_regions)} cells, "
            f"expected {nx_gt * ny_gt}."
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
        area = rect_area(rect)
        if (
            rect.xmin < domain.xmin
            or rect.xmax > domain.xmax
            or rect.ymin < domain.ymin
            or rect.ymax > domain.ymax
        ):
            classification[uid] = ("fail", area)
            continue

        x_range = _fixed_index_range(
            x_edges,
            rect.xmin,
            rect.xmax,
        )
        y_range = _fixed_index_range(
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


def compute_v3_recall(
    absys,
    verified,
    gt_reach_regions,
    *,
    initial_domain: Rect = INITIAL_DOMAIN,
    domain: Rect = DOMAIN,
):
    """Compute mountain-car-v3's initial-domain volume recall."""
    classification = classify_partition_with_v3_ground_truth(
        absys,
        gt_reach_regions,
        domain=domain,
    )

    def is_initial(rect: Rect) -> bool:
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
    gt_goal_uids = {
        uid
        for uid, (label, _) in classification.items()
        if label == "goal" and uid in initial_uids
    }
    verified_goal_uids = gt_goal_uids & set(verified)
    gt_goal_volume = sum(
        classification[uid][1]
        for uid in gt_goal_uids
    )
    verified_goal_volume = sum(
        classification[uid][1]
        for uid in verified_goal_uids
    )
    recall = (
        verified_goal_volume / gt_goal_volume
        if gt_goal_volume
        else float("nan")
    )

    label_counts = {}
    for label in gt_reach_regions.values():
        label_counts[label] = label_counts.get(label, 0) + 1

    print(
        "Recall "
        f"(mountain-car-v3) = {recall:.4f} "
        f"[{verified_goal_volume:.12g}/{gt_goal_volume:.12g}]",
        flush=True,
    )
    return {
        "recall": recall,
        "ground_truth_goal_volume": gt_goal_volume,
        "verified_ground_truth_goal_volume": verified_goal_volume,
        "ground_truth_goal_partition_cells": len(gt_goal_uids),
        "verified_ground_truth_goal_partition_cells": len(
            verified_goal_uids
        ),
        "initial_partition_cells": len(initial_uids),
        "gt_nx": max(int(key[0]) for key in gt_reach_regions) + 1,
        "gt_ny": max(int(key[1]) for key in gt_reach_regions) + 1,
        "gt_fixed_grid_label_counts": label_counts,
    }


def build_abstraction(nx: int, ny: int):
    return mc.build_abstraction(nx=nx, ny=ny)

