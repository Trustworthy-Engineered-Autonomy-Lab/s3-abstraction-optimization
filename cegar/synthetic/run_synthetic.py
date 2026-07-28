"""
run_synthetic.py

Synthetic benchmark using the generic CEGAR engine.

Workflow
--------
1. Build abstraction.
2. Compute initial abstract states.
3. Run generic run_cegar().
4. Compute verified set.
5. Evaluate against ground truth.
6. Save model.
7. Visualize.
"""

from __future__ import annotations

import os
import pickle
import time

from abstraction import compute_verified_set_via_fixpoint
from cegar_loop import run_cegar

from synthetic_benchmark import (
    build_abstraction,
    ground_truth_safe_set,
    compute_tpr,
    compute_recall,
    visualize_classification,
    INITIAL_DOMAIN,
)

###############################################################################
# Configuration
###############################################################################

#
# Set these to match the experiment you are running.
#
NX = 90
NY = 90

MAX_CEGAR_ITERS = 200
MAX_REFINE_DEPTH = 15

MODEL_DIR = "models"

###############################################################################
# Helpers
###############################################################################


def compute_initial_states(absys):

    init = set()

    for uid, node in absys.part.leaves.items():

        r = node.rect

        if (
            r.xmax < INITIAL_DOMAIN.xmin
            or r.xmin > INITIAL_DOMAIN.xmax
            or r.ymax < INITIAL_DOMAIN.ymin
            or r.ymin > INITIAL_DOMAIN.ymax
        ):
            continue

        init.add(uid)

    return init


def save_model(absys, filename):

    os.makedirs(MODEL_DIR, exist_ok=True)

    path = os.path.join(MODEL_DIR, filename)

    with open(path, "wb") as f:
        pickle.dump(absys, f)

    print(f"\nSaved abstraction to {path}")


###############################################################################
# Main
###############################################################################


def main():

    print("=" * 70)
    print("Synthetic Benchmark (True CEGAR)")
    print("=" * 70)

    ###########################################################################
    # Build abstraction
    ###########################################################################

    t0 = time.time()

    absys, domain = build_abstraction(
        nx=NX,
        ny=NY,
    )

    build_time = time.time() - t0

    print(f"Initial cells : {len(absys.part.leaves)}")
    print(f"Build time    : {build_time:.2f} sec")

    ###########################################################################
    # Initial states
    ###########################################################################

    init_states = set(absys.part.leaves.keys())

    print(f"Initial states: {len(init_states)}")

    ###########################################################################
    # Run CEGAR
    ###########################################################################

    print()
    print("=" * 70)
    print("Running generic CEGAR")
    print("=" * 70)

    result = run_cegar(
        absys,
        init_states,
        phi="(!unsafe) U goal",
        action="step",
        max_iters=MAX_CEGAR_ITERS,
        max_refine_depth=MAX_REFINE_DEPTH,
        split_mode="auto",
        verbose=True,
    )
        ###########################################################################
    # CEGAR summary
    ###########################################################################

    print()
    print("=" * 70)
    print("CEGAR Summary")
    print("=" * 70)

    print(f"Verified          : {result.verified}")
    print(f"Iterations        : {result.iterations}")
    print(f"Refinements       : {result.refinements}")
    print(f"Ignored CEX       : {result.ignored_counterexamples}")

    if result.last_cex is not None:

        prefix, cycle = result.last_cex

        print(f"Last prefix length : {len(prefix)}")
        print(f"Last cycle length  : {len(cycle)}")

    ###########################################################################
    # Compute final winning region
    ###########################################################################

    print()
    print("=" * 70)
    print("Computing verified set")
    print("=" * 70)

    t0 = time.time()

    verified = compute_verified_set_via_fixpoint(absys)

    verify_time = time.time() - t0

    print(f"Verified cells : {len(verified)}")
    print(f"Verification time : {verify_time:.2f} sec")

    ###########################################################################
    # Save abstraction model
    ###########################################################################

    filename = f"synthetic_{NX}x{NY}.pkl"

    save_model(
        absys,
        filename,
    )

    ###########################################################################
    # Ground truth
    ###########################################################################

    print()
    print("=" * 70)
    print("Building ground truth")
    print("=" * 70)

    gt_safe = ground_truth_safe_set()
        ###########################################################################
    # Metrics
    ###########################################################################

    print()
    print("=" * 70)
    print("Computing metrics")
    print("=" * 70)

    tpr = compute_tpr(
        absys,
        verified,
        gt_safe,
    )

    recall = compute_recall(
        absys,
        verified,
        gt_safe,
    )

    ###########################################################################
    # Final results
    ###########################################################################

    print()
    print("=" * 70)
    print("Final Results")
    print("=" * 70)

    print(f"Initial grid        : {NX} x {NY}")
    print(f"Initial cells       : {NX * NY}")
    print(f"Final cells         : {len(absys.part.leaves)}")

    print()

    print(f"Verified cells      : {len(verified)}")
    print(f"CEGAR iterations    : {result.iterations}")
    print(f"Refinements         : {result.refinements}")

    print()

    print(f"TPR                 : {tpr:.4f}")
    print(f"Recall              : {recall:.4f}")

    print()

    print(f"Build time          : {build_time:.2f} sec")
    print(f"Verification time   : {verify_time:.2f} sec")

    ###########################################################################
    # Save statistics
    ###########################################################################

    stats = {
        "nx": NX,
        "ny": NY,
        "initial_cells": NX * NY,
        "final_cells": len(absys.part.leaves),
        "verified_cells": len(verified),
        "iterations": result.iterations,
        "refinements": result.refinements,
        "ignored_counterexamples": result.ignored_counterexamples,
        "tpr": tpr,
        "recall": recall,
        "build_time": build_time,
        "verification_time": verify_time,
    }

    os.makedirs(MODEL_DIR, exist_ok=True)

    stats_path = os.path.join(
        MODEL_DIR,
        f"synthetic_{NX}x{NY}_stats.pkl",
    )

    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)

    print(f"\nSaved statistics to {stats_path}")
        ###########################################################################
    # Visualization
    ###########################################################################

    print()
    print("=" * 70)
    print("Generating visualization")
    print("=" * 70)

    figure_name = (
        f"synthetic_cegar_{NX}x{NY}.png"
    )

    visualize_classification(
        absys,
        verified,
        title=(
            f"Synthetic Benchmark "
            f"({NX}x{NY} True CEGAR)"
        ),
        save_path=figure_name,
    )

    print(f"Saved figure: {figure_name}")


###############################################################################
# Entry point
###############################################################################

if __name__ == "__main__":
    main()