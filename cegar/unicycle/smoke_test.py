#!/usr/bin/env python3
"""Build or analyze a Unicycle CEGAR abstraction with a small CLI."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from evaluate_saved_abstractions import (
    DEFAULT_GT,
    evaluate_checkpoint,
)
from gt_cache import default_gt_cache_path, load_or_build_gt_cache
from refine_whole_space_pi import main as run_cegar


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_DIR = SCRIPT_DIR / "artifacts"
DEFAULT_SHAPE = 5
DEFAULT_GT_STEPS = 10
DEFAULT_TIME_LIMIT_SEC = 5.0


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def default_output(shape: int) -> Path:
    return DEFAULT_ARTIFACT_DIR / (
        f"unicycle_cegar_{shape}x{shape}x{shape}.pkl"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test Unicycle CEGAR by either building/refining a fresh "
            "cubic abstraction or loading a saved model, then optionally "
            "running the reference pyModelChecking analysis."
        )
    )
    parser.add_argument(
        "--shape",
        type=positive_int,
        default=DEFAULT_SHAPE,
        help=(
            "Initial grid resolution on every axis when building "
            f"(default: {DEFAULT_SHAPE})."
        ),
    )
    parser.add_argument(
        "--load-model",
        type=Path,
        help="Load and analyze this checkpoint instead of building a model.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Checkpoint path for a new model (default: "
            "artifacts/unicycle_cegar_NxNxN.pkl)."
        ),
    )
    parser.add_argument(
        "--gt-cache",
        type=Path,
        help=(
            "Matching native CEGAR GT cache used during a build. The default "
            "is artifacts/cache/gt_safe_unicycle_NxNxN_stepsK.pkl."
        ),
    )
    parser.add_argument(
        "--gt-steps",
        type=positive_int,
        default=DEFAULT_GT_STEPS,
        help=f"Simulation steps for the native CEGAR GT cache (default: {DEFAULT_GT_STEPS}).",
    )
    parser.add_argument(
        "--analysis-gt",
        type=Path,
        default=DEFAULT_GT,
        help=(
            "Fixed unicycle-taylor reachability labels used for final "
            "pyModelChecking recall (default: artifacts/cache/reach.pkl)."
        ),
    )
    parser.add_argument(
        "--time-limit-sec",
        type=nonnegative_float,
        default=DEFAULT_TIME_LIMIT_SEC,
        help=f"CEGAR refinement time budget (default: {DEFAULT_TIME_LIMIT_SEC:g}).",
    )
    parser.add_argument(
        "--max-total-states",
        type=positive_int,
        help="Optional live-cell limit; otherwise the production runner uses (N+10)^3.",
    )
    parser.add_argument(
        "--checkpoint-interval-sec",
        type=nonnegative_float,
        default=900.0,
        help="Periodic checkpoint interval during refinement (default: 900).",
    )
    parser.add_argument(
        "--max-iters",
        "--max-iters-per-cell",
        dest="max_iters",
        type=positive_int,
        default=150,
        help="Maximum CEGAR iterations per candidate cell (default: 150).",
    )
    parser.add_argument(
        "--max-depth",
        "--max-refine-depth",
        dest="max_depth",
        type=positive_int,
        default=40,
        help="Maximum refinement depth (default: 40).",
    )
    parser.add_argument(
        "--min-size",
        "--min-cell-size",
        dest="min_size",
        type=positive_float,
        default=0.001,
        help="Minimum spatial and heading cell extent (default: 0.001).",
    )
    parser.add_argument(
        "--split",
        "--split-mode",
        dest="split",
        choices=("auto", "xy", "xyz"),
        default="auto",
        help="Cell split mode (default: auto).",
    )
    parser.add_argument(
        "--backend",
        "--counterexample-backend",
        dest="backend",
        choices=("graph", "auto", "spot"),
        default="graph",
        help="Counterexample backend (default: graph).",
    )
    parser.add_argument(
        "--fresh",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build from scratch instead of resuming the output checkpoint.",
    )
    parser.add_argument(
        "--analyze",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run pyModelChecking and reference recall after loading/building.",
    )
    parser.add_argument(
        "--write-results",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write the *.pymodelchecking.json analysis result.",
    )
    args = parser.parse_args(argv)
    if args.load_model is not None and not args.analyze:
        parser.error("--load-model cannot be combined with --no-analyze")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    analysis_gt = args.analysis_gt.resolve()
    if args.analyze and not analysis_gt.exists():
        raise FileNotFoundError(
            "Reference analysis GT does not exist: "
            f"{analysis_gt}. Expected the case-local cache at "
            "artifacts/cache/reach.pkl, or pass --analysis-gt."
        )

    if args.load_model is not None:
        checkpoint = args.load_model.resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Saved model does not exist: {checkpoint}"
            )
        print(f"[SMOKE] loading saved model: {checkpoint}", flush=True)
    else:
        checkpoint = (
            args.output.resolve()
            if args.output is not None
            else default_output(args.shape)
        )
        native_gt = (
            args.gt_cache.resolve()
            if args.gt_cache is not None
            else default_gt_cache_path(
                args.shape,
                args.shape,
                args.shape,
                args.gt_steps,
            )
        )
        print(
            f"[SMOKE] ensuring native {args.shape}x{args.shape}x{args.shape} "
            f"GT cache: {native_gt}",
            flush=True,
        )
        load_or_build_gt_cache(
            native_gt,
            nx_gt=args.shape,
            ny_gt=args.shape,
            nz_gt=args.shape,
            max_steps=args.gt_steps,
        )

        runner_args = [
            "--grid-size",
            str(args.shape),
            "--checkpoint",
            str(checkpoint),
            "--gt-cache",
            str(native_gt),
            "--gt-max-steps",
            str(args.gt_steps),
            "--time-limit-sec",
            str(args.time_limit_sec),
            "--checkpoint-interval-sec",
            str(args.checkpoint_interval_sec),
            "--max-iters-per-cell",
            str(args.max_iters),
            "--max-refine-depth",
            str(args.max_depth),
            "--min-cell-size",
            str(args.min_size),
            "--min-cell-theta",
            str(args.min_size),
            "--split-mode",
            args.split,
            "--counterexample-backend",
            args.backend,
        ]
        if args.max_total_states is not None:
            runner_args.extend(
                ["--max-total-states", str(args.max_total_states)]
            )
        if args.fresh:
            runner_args.append("--fresh")

        print(
            f"[SMOKE] building/refining model: {checkpoint}",
            flush=True,
        )
        exit_code = run_cegar(runner_args)
        if exit_code:
            return int(exit_code)

    if args.analyze:
        print(
            f"[SMOKE] running pyModelChecking analysis with {analysis_gt}",
            flush=True,
        )
        evaluate_checkpoint(
            checkpoint,
            analysis_gt,
            write=args.write_results,
        )
    else:
        print("[SMOKE] analysis disabled.", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
