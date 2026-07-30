#!/usr/bin/env python3
"""Small unified CLI for building or analyzing spiral CEGAR models."""
from __future__ import annotations

import argparse
from pathlib import Path

from evaluate_saved_abstractions import evaluate_checkpoint
from run_synthetic import main as run_synthetic


SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = SCRIPT_DIR / "artifacts"
DEFAULT_GT_CACHE = ARTIFACT_DIR / "cache" / "reach.pkl"

SHAPE = 10
TIME_LIMIT_SEC = 5.0
CHECKPOINT_INTERVAL_SEC = 900.0
MAX_ITERS_PER_CELL = 150
MAX_REFINE_DEPTH = 40
MIN_CELL_SIZE = 1e-6


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Build, refine, save, and optionally analyze a spiral CEGAR "
            "model, or load and analyze an existing saved model."
        )
    )
    parser.add_argument(
        "--load-model",
        type=Path,
        help=(
            "Analyze this saved CEGAR checkpoint instead of building a model."
        ),
    )
    parser.add_argument(
        "--shape",
        type=positive_int,
        default=SHAPE,
        help=f"Initial NxN grid size for a new model (default: {SHAPE}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "New-model checkpoint path. Defaults to "
            "artifacts/synthetic_cegar_NxN.pkl."
        ),
    )
    parser.add_argument(
        "--time-limit-sec",
        type=nonnegative_float,
        default=TIME_LIMIT_SEC,
        help=(
            "CEGAR refinement time limit in seconds "
            f"(default: {TIME_LIMIT_SEC:g})."
        ),
    )
    parser.add_argument(
        "--max-total-states",
        type=positive_int,
        help="Live-cell cap; the runner default is (shape + 10)^2.",
    )
    parser.add_argument(
        "--checkpoint-interval-sec",
        type=nonnegative_float,
        default=CHECKPOINT_INTERVAL_SEC,
    )
    parser.add_argument(
        "--max-iters",
        "--max-iters-per-cell",
        dest="max_iters_per_cell",
        type=positive_int,
        default=MAX_ITERS_PER_CELL,
    )
    parser.add_argument(
        "--max-depth",
        "--max-refine-depth",
        dest="max_refine_depth",
        type=positive_int,
        default=MAX_REFINE_DEPTH,
    )
    parser.add_argument(
        "--min-size",
        "--min-cell-size",
        dest="min_cell_size",
        type=positive_float,
        default=MIN_CELL_SIZE,
    )
    parser.add_argument(
        "--split",
        "--split-mode",
        dest="split_mode",
        choices=("auto", "xy"),
        default="auto",
    )
    parser.add_argument(
        "--backend",
        "--counterexample-backend",
        dest="counterexample_backend",
        choices=("graph", "auto", "spot"),
        default="graph",
    )
    parser.add_argument(
        "--fresh",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Build from a fresh uniform model. Use --no-fresh to resume the "
            "output checkpoint when it exists."
        ),
    )
    parser.add_argument(
        "--analyze",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run pyModelChecking and recall analysis after a new model is saved.",
    )
    parser.add_argument(
        "--write-results",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write the pyModelChecking result JSON beside the checkpoint.",
    )
    parser.add_argument(
        "--gt-cache",
        type=Path,
        default=DEFAULT_GT_CACHE,
        help="Spiral fixed-grid reachability reference.",
    )
    args = parser.parse_args(argv)
    if args.load_model is not None and not args.analyze:
        parser.error("--load-model cannot be combined with --no-analyze")
    return args


def default_output_path(shape: int) -> Path:
    return ARTIFACT_DIR / f"synthetic_cegar_{shape}x{shape}.pkl"


def main(argv=None) -> int:
    args = parse_args(argv)
    gt_cache = args.gt_cache.resolve()

    if args.load_model is not None:
        evaluate_checkpoint(
            args.load_model.resolve(),
            gt_cache,
            args.write_results,
        )
        return 0

    output = (
        args.output.resolve()
        if args.output is not None
        else default_output_path(args.shape)
    )
    runner_args = [
        "--shape",
        str(args.shape),
        "--checkpoint",
        str(output),
        "--time-limit-sec",
        str(args.time_limit_sec),
        "--checkpoint-interval-sec",
        str(args.checkpoint_interval_sec),
        "--max-iters-per-cell",
        str(args.max_iters_per_cell),
        "--max-refine-depth",
        str(args.max_refine_depth),
        "--min-cell-size",
        str(args.min_cell_size),
        "--split-mode",
        args.split_mode,
        "--counterexample-backend",
        args.counterexample_backend,
        "--gt-cache",
        str(gt_cache),
    ]
    if args.max_total_states is not None:
        runner_args.extend(
            ["--max-total-states", str(args.max_total_states)]
        )
    if args.fresh:
        runner_args.append("--fresh")

    status = run_synthetic(runner_args)
    if status:
        return status

    if args.analyze:
        evaluate_checkpoint(output, gt_cache, args.write_results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
