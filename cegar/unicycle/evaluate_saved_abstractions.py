#!/usr/bin/env python3
"""Run unicycle-taylor model checking and recall on saved CEGAR models."""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pyModelChecking as pmc
import pyModelChecking.CTL as CTL

from main import INIT_DOMAIN_LB, INIT_DOMAIN_UB
from refine_whole_space_pi import load_model_checkpoint


SCRIPT_DIR = Path(__file__).resolve().parent
PROPERTY = "A (safe U goal)"
DEFAULT_GT = SCRIPT_DIR / "artifacts" / "cache" / "reach.pkl"
DEFAULT_CHECKPOINTS = (
    SCRIPT_DIR / "artifacts" / "unicycle_cegar_40x40x40.pkl",
    SCRIPT_DIR / "artifacts" / "unicycle_cegar_90x90x90.pkl",
)


def is_initial(rect) -> bool:
    """Match unicycle-taylor's side='right' initial-cell convention."""
    return (
        rect.xmax > float(INIT_DOMAIN_LB[0])
        and rect.xmin <= float(INIT_DOMAIN_UB[0])
        and rect.ymax > float(INIT_DOMAIN_LB[1])
        and rect.ymin <= float(INIT_DOMAIN_UB[1])
        and rect.zmax > float(INIT_DOMAIN_LB[2])
        and rect.zmin <= float(INIT_DOMAIN_UB[2])
    )


def build_kripke(absys):
    leaves = absys.part.leaves
    states = set(leaves)
    states.add(absys.OUT_UID)
    initial_states = {
        uid for uid, node in leaves.items() if is_initial(node.rect)
    }

    edge_count = [0]

    def transitions():
        for source in states:
            destinations = set()
            for values in absys.tr.succ.get(source, {}).values():
                destinations.update(values)
            if not destinations:
                destinations.add(source)
            invalid = destinations - states
            if invalid:
                raise ValueError(
                    f"Transition source {source} has invalid targets: "
                    f"{sorted(invalid)[:10]}"
                )
            for target in destinations:
                edge_count[0] += 1
                yield source, target

    started = time.perf_counter()
    kripke = pmc.Kripke(
        S=states,
        S0=initial_states,
        R=transitions(),
        L=None,
    )
    labels = kripke.labelling_function()
    labels[absys.OUT_UID] = {"fail"}
    for uid, node in leaves.items():
        aps = set(absys.ap_labeler(node.rect))
        if "goal" in aps:
            labels[uid] = {"goal"}
        elif "unsafe" in aps:
            labels[uid] = {"fail"}
        else:
            labels[uid] = {"safe"}
    build_elapsed = time.perf_counter() - started
    return kripke, initial_states, edge_count[0], build_elapsed


def fixed_index_range(edges, lo, hi):
    """Exact range convention from unicycle-taylor verification_tools."""
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


def compute_unicycle_taylor_recall(
    absys,
    satisfying_initial,
    initial_states,
    gt_reach_regions,
    domain,
) -> dict:
    if not gt_reach_regions:
        raise ValueError("Ground-truth reach-region dictionary is empty.")
    if not all(
        isinstance(key, tuple) and len(key) == 3
        for key in gt_reach_regions
    ):
        raise ValueError("Expected ground-truth keys of form (i, j, k).")

    nx = max(int(key[0]) for key in gt_reach_regions) + 1
    ny = max(int(key[1]) for key in gt_reach_regions) + 1
    nz = max(int(key[2]) for key in gt_reach_regions) + 1
    if len(gt_reach_regions) != nx * ny * nz:
        raise ValueError(
            "Ground-truth grid is incomplete: "
            f"found {len(gt_reach_regions)}, expected {nx * ny * nz}."
        )

    x_edges = np.linspace(domain.xmin, domain.xmax, nx + 1)
    y_edges = np.linspace(domain.ymin, domain.ymax, ny + 1)
    z_edges = np.linspace(domain.zmin, domain.zmax, nz + 1)
    goal_cells = {
        (int(key[0]), int(key[1]), int(key[2]))
        for key, label in gt_reach_regions.items()
        if label == "goal"
    }

    gt_goal_volume = 0.0
    verified_goal_volume = 0.0
    gt_goal_count = 0
    verified_goal_count = 0
    satisfying_initial = set(satisfying_initial)

    for uid in initial_states:
        rect = absys.part.leaves[uid].rect
        if (
            rect.xmin < domain.xmin
            or rect.xmax > domain.xmax
            or rect.ymin < domain.ymin
            or rect.ymax > domain.ymax
            or rect.zmin < domain.zmin
            or rect.zmax > domain.zmax
        ):
            continue

        x_range = fixed_index_range(x_edges, rect.xmin, rect.xmax)
        y_range = fixed_index_range(y_edges, rect.ymin, rect.ymax)
        z_range = fixed_index_range(z_edges, rect.zmin, rect.zmax)
        if x_range is None or y_range is None or z_range is None:
            continue

        all_goal = all(
            (i, j, k) in goal_cells
            for i in range(x_range[0], x_range[1] + 1)
            for j in range(y_range[0], y_range[1] + 1)
            for k in range(z_range[0], z_range[1] + 1)
        )
        if not all_goal:
            continue

        volume = float(
            (rect.xmax - rect.xmin)
            * (rect.ymax - rect.ymin)
            * (rect.zmax - rect.zmin)
        )
        gt_goal_volume += volume
        gt_goal_count += 1
        if uid in satisfying_initial:
            verified_goal_volume += volume
            verified_goal_count += 1

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
        f"(unicycle-taylor) = {recall:.4f} "
        f"[{verified_goal_volume:.12g}/{gt_goal_volume:.12g}]",
        flush=True,
    )
    return {
        "recall": recall,
        "ground_truth_goal_volume": gt_goal_volume,
        "verified_ground_truth_goal_volume": verified_goal_volume,
        "ground_truth_goal_partition_cells": gt_goal_count,
        "verified_ground_truth_goal_partition_cells": verified_goal_count,
        "initial_partition_cells": len(initial_states),
        "gt_nx": nx,
        "gt_ny": ny,
        "gt_ntheta": nz,
        "gt_fixed_grid_label_counts": label_counts,
    }


def evaluate_checkpoint(checkpoint: Path, gt_path: Path, write: bool) -> dict:
    checkpoint = checkpoint.resolve()
    absys, domain, _ = load_model_checkpoint(checkpoint)
    kripke, initial_states, edge_count, build_elapsed = build_kripke(absys)

    started = time.perf_counter()
    satisfying = set(CTL.modelcheck(kripke, PROPERTY))
    model_checking_elapsed = time.perf_counter() - started
    satisfying_initial = satisfying & initial_states

    with gt_path.open("rb") as stream:
        gt_reach_regions = pickle.load(stream)
    recall = compute_unicycle_taylor_recall(
        absys,
        satisfying_initial,
        initial_states,
        gt_reach_regions,
        domain,
    )

    result = {
        "case_study": "unicycle",
        "reference": "unicycle-taylor",
        "checkpoint": str(checkpoint),
        "property": PROPERTY,
        "states": len(kripke.states()),
        "initial_states": len(initial_states),
        "transitions": edge_count,
        "kripke_construction_time_sec": build_elapsed,
        "model_checking_time_sec": model_checking_elapsed,
        "satisfying_initial_states": len(satisfying_initial),
        "recall": recall["recall"],
        "recall_details": recall,
        "ground_truth": str(gt_path),
    }

    print(
        f"[UNICYCLE] {checkpoint.name}: "
        f"model_checking_time_sec={model_checking_elapsed:.6f}, "
        f"recall={recall['recall']:.6f}",
        flush=True,
    )
    if write:
        output = checkpoint.with_suffix(".pymodelchecking.json")
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
        print(f"[WRITE] {output}", flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Load saved Unicycle CEGAR abstractions, run pyModelChecking "
            "with unicycle-taylor's property, time it, and compute "
            "unicycle-taylor recall."
        )
    )
    parser.add_argument("checkpoints", nargs="*", type=Path)
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    checkpoints = args.checkpoints or list(DEFAULT_CHECKPOINTS)
    gt_path = args.gt.resolve()
    for checkpoint in checkpoints:
        evaluate_checkpoint(checkpoint, gt_path, not args.no_write)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
