# =====================================================================
# Description: user-friendy wrapper to for building or analyzing
# Mountain Car CEGAR models.
# =====================================================================


# =====================================================================
# Libraries
# =====================================================================

from __future__ import annotations
import argparse
from pathlib import Path


# =====================================================================
# User-defined settings
# =====================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = SCRIPT_DIR / "artifacts"
DEFAULT_GT_CACHE = ARTIFACT_DIR / "cache" / "reach.pkl"

DEFAULT_SHAPE = 10
DEFAULT_TIME_LIMIT_SEC = 5.0
DEFAULT_CHECKPOINT_INTERVAL_SEC = 60.0
DEFAULT_MAX_ITERS = 10
DEFAULT_MAX_DEPTH = 25
DEFAULT_MIN_SIZE = 1e-5


def default_output_path(shape: int) -> Path:
    return ARTIFACT_DIR / f"mountain_car_cegar_{shape}x{shape}.pkl"

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Build, refine, save, and optionally analyze a Mountain Car "
            "CEGAR abstraction, or analyze an existing saved abstraction."
        )
    )
    parser.add_argument(
        "--shape",
        type=int,
        default=DEFAULT_SHAPE,
        help="Uniform initial grid size N for an N x N model (default: 10).",
    )
    parser.add_argument(
        "--load-model",
        type=Path,
        help=(
            "Load and analyze this saved model instead of building a new "
            "one."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Build output checkpoint. The default is "
            "artifacts/mountain_car_cegar_NxN.pkl."
        ),
    )
    parser.add_argument(
        "--gt-cache",
        type=Path,
        default=DEFAULT_GT_CACHE,
        help="Ground-truth reachability cache used to compute recall.",
    )
    parser.add_argument(
        "--time-limit-sec",
        type=float,
        default=DEFAULT_TIME_LIMIT_SEC,
        help="CEGAR time limit in seconds (default: 5).",
    )
    parser.add_argument(
        "--max-total-states",
        type=int,
        help=(
            "Optional live-state cap. By default the runner uses "
            "(shape + 10)^2."
        ),
    )
    parser.add_argument(
        "--checkpoint-interval-sec",
        type=float,
        default=DEFAULT_CHECKPOINT_INTERVAL_SEC,
        help="Seconds between periodic checkpoints (default: 60).",
    )
    parser.add_argument(
        "--max-iters",
        "--max-iters-per-cell",
        dest="max_iters",
        type=int,
        default=DEFAULT_MAX_ITERS,
        help="Maximum CEGAR iterations per candidate cell (default: 10).",
    )
    parser.add_argument(
        "--max-depth",
        "--max-refine-depth",
        dest="max_depth",
        type=int,
        default=DEFAULT_MAX_DEPTH,
        help="Maximum refinement depth (default: 25).",
    )
    parser.add_argument(
        "--min-size",
        "--min-cell-size",
        dest="min_size",
        type=float,
        default=DEFAULT_MIN_SIZE,
        help="Minimum refinable cell width/height (default: 1e-5).",
    )
    parser.add_argument(
        "--split",
        "--split-policy",
        "--split-mode",
        dest="split_policy",
        choices=("auto", "xy"),
        default="auto",
        help="Existing CEGAR split policy (default: auto).",
    )
    parser.add_argument(
        "--backend",
        "--counterexample-backend",
        dest="backend",
        choices=("graph", "auto", "spot"),
        default="graph",
        help="Existing counterexample backend (default: graph).",
    )
    parser.add_argument(
        "--fresh",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Start from a new uniform abstraction. Use --no-fresh to resume "
            "the output checkpoint."
        ),
    )
    parser.add_argument(
        "--analyze",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run pyModelChecking and compute recall after building "
            "(default: true)."
        ),
    )
    parser.add_argument(
        "--write-results",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write the analysis JSON beside the model (default: true).",
    )
    args = parser.parse_args(argv)

    if args.shape <= 0:
        parser.error("--shape must be positive")
    if args.time_limit_sec < 0:
        parser.error("--time-limit-sec must be nonnegative")
    if args.max_total_states is not None and args.max_total_states <= 0:
        parser.error("--max-total-states must be positive")
    if args.checkpoint_interval_sec < 0:
        parser.error("--checkpoint-interval-sec must be nonnegative")
    if args.max_iters <= 0:
        parser.error("--max-iters must be positive")
    if args.max_depth <= 0:
        parser.error("--max-depth must be positive")
    if args.min_size <= 0:
        parser.error("--min-size must be positive")
    if args.load_model is not None and not args.analyze:
        parser.error("--load-model cannot be combined with --no-analyze")
    return args


def analyze_model(model_path: Path, gt_cache: Path, *, write: bool) -> dict:
    # Import lazily so `smoke_test.py --help` does not load the DDPG.
    from evaluate_saved_abstractions import evaluate_checkpoint

    return evaluate_checkpoint(
        model_path.resolve(),
        gt_cache.resolve(),
        write,
    )


# =====================================================================
# Main
# =====================================================================

def main(argv=None) -> int:
    args = parse_args(argv)
    gt_cache = args.gt_cache.resolve()

    if args.load_model is not None:
        analyze_model(
            args.load_model,
            gt_cache,
            write=args.write_results,
        )
        return 0

    output_path = (
        args.output.resolve()
        if args.output is not None
        else default_output_path(args.shape)
    )
    if not args.fresh and not output_path.exists():
        raise FileNotFoundError(
            "Cannot resume a missing output checkpoint: "
            f"{output_path}"
        )

    runner_argv = [
        "--grid-size",
        str(args.shape),
        "--checkpoint",
        str(output_path),
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
        "--split-mode",
        args.split_policy,
        "--counterexample-backend",
        args.backend,
        "--gt-reach-regions",
        str(gt_cache),
        "--fresh" if args.fresh else "--resume-existing",
    ]
    if args.max_total_states is not None:
        runner_argv.extend(
            ["--max-total-states", str(args.max_total_states)]
        )

    # Delegate to the production runner rather than duplicating CEGAR logic.
    from run_mountain_car import main as run_mountain_car

    return_code = run_mountain_car(runner_argv)
    if return_code:
        return return_code

    if args.analyze:
        analyze_model(
            output_path,
            gt_cache,
            write=args.write_results,
        )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
