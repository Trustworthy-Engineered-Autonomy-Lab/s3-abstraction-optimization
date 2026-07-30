#!/usr/bin/env python3
"""Run mountain-car-v3 model checking and recall on saved CEGAR models."""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import pyModelChecking as pmc
import pyModelChecking.CTL as CTL

from mountain_car_benchmark import DOMAIN, compute_v3_recall
from run_mountain_car import load_model_checkpoint


SCRIPT_DIR = Path(__file__).resolve().parent
PROPERTY = "A F goal"
DEFAULT_GT = SCRIPT_DIR / "artifacts" / "cache" / "reach.pkl"
DEFAULT_CHECKPOINTS = (
    SCRIPT_DIR / "artifacts" / "mountain_car_cegar_60x60.pkl",
    SCRIPT_DIR / "artifacts" / "mountain_car_cegar_90x90.pkl",
)


def build_kripke(absys):
    leaves = absys.part.leaves
    states = set(leaves)
    states.add(absys.OUT_UID)
    initial_states = set(leaves)

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
        labels[uid] = {"goal"} if "goal" in aps else {"safe"}
    build_elapsed = time.perf_counter() - started
    return kripke, initial_states, edge_count[0], build_elapsed


def evaluate_checkpoint(checkpoint: Path, gt_path: Path, write: bool) -> dict:
    checkpoint = checkpoint.resolve()
    absys, _, _ = load_model_checkpoint(checkpoint)
    kripke, initial_states, edge_count, build_elapsed = build_kripke(absys)

    started = time.perf_counter()
    satisfying = set(CTL.modelcheck(kripke, PROPERTY))
    model_checking_elapsed = time.perf_counter() - started
    satisfying_initial = satisfying & initial_states

    with gt_path.open("rb") as stream:
        gt_reach_regions = pickle.load(stream)
    recall = compute_v3_recall(
        absys,
        satisfying_initial,
        gt_reach_regions,
        initial_domain=DOMAIN,
        domain=DOMAIN,
    )

    result = {
        "case_study": "mountain-car",
        "reference": "mountain-car-v3",
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
        f"[MOUNTAIN-CAR] {checkpoint.name}: "
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
            "Load saved Mountain Car CEGAR abstractions, run "
            "pyModelChecking with mountain-car-v3's property, time it, "
            "and compute v3 recall."
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
