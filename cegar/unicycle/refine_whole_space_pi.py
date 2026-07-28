#!/usr/bin/env python3
"""Timed, resumable CEGAR runner for the unicycle abstraction."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import signal
import sys
import time
from pathlib import Path
from typing import Any, Optional, Tuple

# Windows PowerShell commonly gives Python a cp1252 stdout.  The older
# reporting code uses Unicode separators, so replace unsupported characters
# instead of letting a successful multi-hour run crash while printing recall.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")

from abstraction import Abstraction, Rect
from gt_cache import load_gt_cache
from main import (
    INIT_DOMAIN_LB,
    INIT_DOMAIN_UB,
    build_abstraction,
)
from refine_whole_space import (
    RegionClassification,
    classify_all_leaves_once,
    compute_metrics,
    compute_recall,
    compute_tpr,
    refine_one_round,
    save_model_checkpoint,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_DIR = SCRIPT_DIR / "artifacts"
PHI = "(!unsafe) U goal"
STOP_REQUESTED = False


def _request_stop(signum, frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(
        f"\n[STOP] received signal {signum}; stopping at the next safe "
        "iteration boundary.",
        flush=True,
    )


for _signal_name in ("SIGTERM", "SIGINT", "SIGUSR1"):
    _signal = getattr(signal, _signal_name, None)
    if _signal is not None:
        try:
            signal.signal(_signal, _request_stop)
        except (OSError, RuntimeError, ValueError):
            pass


def _default_checkpoint(grid_size: int) -> Path:
    return DEFAULT_ARTIFACT_DIR / (
        f"unicycle_cegar_{grid_size}x{grid_size}x{grid_size}.pkl"
    )


def _default_gt_cache(grid_size: int, max_steps: int) -> Path:
    return SCRIPT_DIR / (
        f"gt_safe_unicycle_{grid_size}x{grid_size}x{grid_size}"
        f"_steps{max_steps}.pkl"
    )


def _default_summary(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_suffix(".summary.json")


def load_model_checkpoint(path: str | Path) -> Tuple[Any, Rect, dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint does not exist: {path}")

    with path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or "absys" not in payload:
        raise ValueError(
            f"{path} is not a model checkpoint containing an 'absys' entry."
        )

    absys = payload["absys"]
    domain = payload.get("domain")
    if domain is None:
        from main import X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX

        domain = Rect(X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX)
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {"raw_metadata": repr(metadata)}

    print(f"[LOAD] checkpoint: {path}", flush=True)
    print(
        f"[LOAD] leaves={len(absys.part.leaves)} "
        f"transition_sources={len(absys.tr.succ)}",
        flush=True,
    )
    completed_count = metadata.get(
        "completed_candidate_count",
        len(metadata.get("completed_candidate_uids", ())),
    )
    print(
        f"[LOAD] stage={metadata.get('stage')!r} "
        f"completed_candidates={completed_count}",
        flush=True,
    )
    return absys, domain, metadata


def abstraction_grid_shape(absys: Abstraction) -> Tuple[int, int, int]:
    dyn = absys.dyn_by_action.get("step")
    if (
        dyn is not None
        and hasattr(dyn, "x_edges")
        and hasattr(dyn, "y_edges")
        and hasattr(dyn, "theta_edges")
    ):
        return (
            len(dyn.x_edges) - 1,
            len(dyn.y_edges) - 1,
            len(dyn.theta_edges) - 1,
        )

    roots = absys.part.roots
    x_intervals = {(n.rect.xmin, n.rect.xmax) for n in roots}
    y_intervals = {(n.rect.ymin, n.rect.ymax) for n in roots}
    z_intervals = {(n.rect.zmin, n.rect.zmax) for n in roots}
    return len(x_intervals), len(y_intervals), len(z_intervals)


def transition_stats(absys: Abstraction, *, validate: bool = True) -> dict:
    valid_uids = set(absys.part.leaves)
    valid_uids.add(absys.OUT_UID)
    missing_sources = [
        uid for uid in absys.part.leaves if uid not in absys.tr.succ
    ]
    dangling = 0
    edge_count = 0
    for by_action in absys.tr.succ.values():
        for destinations in by_action.values():
            edge_count += len(destinations)
            if validate:
                dangling += sum(dst not in valid_uids for dst in destinations)

    stats = {
        "leaf_count": len(absys.part.leaves),
        "transition_source_count": len(absys.tr.succ),
        "transition_edge_count": edge_count,
        "missing_transition_sources": len(missing_sources),
        "dangling_transition_edges": dangling,
    }
    if validate and (missing_sources or dangling):
        raise ValueError(
            "Abstraction transition relation is incomplete: "
            f"missing_sources={len(missing_sources)}, dangling_edges={dangling}."
        )
    return stats


def initial_domain_rect() -> Rect:
    return Rect(
        float(INIT_DOMAIN_LB[0]),
        float(INIT_DOMAIN_UB[0]),
        float(INIT_DOMAIN_LB[1]),
        float(INIT_DOMAIN_UB[1]),
        float(INIT_DOMAIN_LB[2]),
        float(INIT_DOMAIN_UB[2]),
    )


def load_matching_gt_cache(path: Path, grid_size: int, max_steps: int) -> dict:
    payload = load_gt_cache(path)
    actual = (
        int(payload["nx_gt"]),
        int(payload["ny_gt"]),
        int(payload["nz_gt"]),
        int(payload["max_steps"]),
    )
    expected = (grid_size, grid_size, grid_size, max_steps)
    if actual != expected:
        raise ValueError(
            f"GT cache {path} describes {actual[0]}x{actual[1]}x{actual[2]}, "
            f"steps={actual[3]}, not the requested "
            f"{grid_size}x{grid_size}x{grid_size}, steps={max_steps}."
        )
    print(
        f"[GT] loaded {len(payload['gt_safe'])} safe voxels from {path}",
        flush=True,
    )
    return payload


def evaluate(
    absys: Abstraction,
    domain: Rect,
    *,
    grid_size: int,
    gt_cache_path: Path,
    gt_max_steps: int,
    classification: Optional[RegionClassification] = None,
) -> Tuple[RegionClassification, dict]:
    if classification is None:
        classification = classify_all_leaves_once(absys, PHI, action="step")

    verified = set(classification.verified)
    refuted = set(classification.refuted)
    unknown = set(classification.unknown)
    metrics = compute_metrics(absys, verified, refuted, unknown)

    gt_payload = load_matching_gt_cache(
        gt_cache_path, grid_size, gt_max_steps
    )
    init_domain = initial_domain_rect()
    print(
        "[EVAL] initial domain: "
        f"x=[{init_domain.xmin},{init_domain.xmax}] "
        f"y=[{init_domain.ymin},{init_domain.ymax}] "
        f"theta=[{init_domain.zmin:.6f},{init_domain.zmax:.6f}]",
        flush=True,
    )
    recall = compute_recall(
        absys,
        verified,
        domain,
        gt_safe=gt_payload["gt_safe"],
        nx_gt=grid_size,
        ny_gt=grid_size,
        nz_gt=grid_size,
        initial_domain=init_domain,
    )
    tpr = compute_tpr(
        absys,
        verified,
        domain,
        gt_safe=gt_payload["gt_safe"],
        nx_gt=grid_size,
        ny_gt=grid_size,
        nz_gt=grid_size,
        initial_domain=init_domain,
    )

    result = {
        **metrics,
        "recall_gt_safe_volume": recall,
        "tpr_current_partition": tpr,
        "gt_safe_voxels": len(gt_payload["gt_safe"]),
        "gt_grid_size": grid_size,
        "gt_max_steps": gt_max_steps,
        **transition_stats(absys),
    }
    return classification, result


def write_summary(path: Path, summary: dict) -> None:
    def json_default(value):
        if isinstance(value, set):
            return sorted(value)
        if isinstance(value, Path):
            return str(value)
        return repr(value)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
            default=json_default,
        ) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)
    print(f"[SUMMARY] wrote {path}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Build/resume a complete unicycle abstraction, run reach-avoid "
            "CEGAR for a bounded time, checkpoint it, and report final recall."
        )
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        choices=(40, 90),
        required=True,
        help="Cubic initial abstraction size (40 or 90).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Output/resume checkpoint path (default is under artifacts/).",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help=(
            "Load this checkpoint if the output checkpoint does not yet "
            "exist; subsequent saves go to --checkpoint."
        ),
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore existing checkpoints and construct a fresh uniform grid.",
    )
    parser.add_argument(
        "--time-limit-sec",
        type=float,
        default=float(os.environ.get("TIME_LIMIT_SEC", 3 * 60 * 60)),
        help="CEGAR refinement budget in seconds (default: 10800).",
    )
    parser.add_argument(
        "--safety-margin-sec",
        type=float,
        default=float(os.environ.get("TIME_LIMIT_SAFETY_MARGIN_SEC", 0)),
        help="Stop this many seconds before the stated limit (default: 0).",
    )
    parser.add_argument(
        "--checkpoint-interval-sec",
        type=float,
        default=float(os.environ.get("CHECKPOINT_INTERVAL_SEC", 15 * 60)),
        help="Periodic checkpoint interval during CEGAR (default: 900).",
    )
    parser.add_argument(
        "--max-total-states",
        type=int,
        help=(
            "Maximum live abstraction cells, excluding OUT. The default is "
            "(grid_size + 10)^3: 125000 for 40 and 1000000 for 90."
        ),
    )
    parser.add_argument(
        "--gt-cache",
        type=Path,
        help="Matching gt_safe cache used only for final evaluation.",
    )
    parser.add_argument(
        "--gt-max-steps",
        type=int,
        default=int(os.environ.get("GT_MAX_STEPS", 10)),
    )
    parser.add_argument(
        "--ordering",
        choices=("largest", "smallest", "random"),
        default=os.environ.get("ORDERING", "largest"),
    )
    parser.add_argument(
        "--rand-seed",
        type=int,
        default=int(os.environ.get("RAND_SEED", 0)),
    )
    parser.add_argument(
        "--split-mode",
        choices=("auto", "xy", "xyz"),
        default=os.environ.get("SPLIT_MODE", "auto"),
    )
    parser.add_argument(
        "--max-iters-per-cell",
        type=int,
        default=int(os.environ.get("MAX_ITERS_PER_CELL", 150)),
    )
    parser.add_argument(
        "--max-refine-depth",
        type=int,
        default=int(os.environ.get("MAX_REFINE_DEPTH", 40)),
    )
    parser.add_argument(
        "--min-cell-size",
        type=float,
        default=float(os.environ.get("MIN_CELL_SIZE", 0.001)),
    )
    parser.add_argument(
        "--min-cell-theta",
        type=float,
        default=float(os.environ.get("MIN_CELL_THETA", 0.001)),
    )
    parser.add_argument(
        "--gc-every",
        type=int,
        default=int(os.environ.get("GC_EVERY", 100)),
    )
    parser.add_argument(
        "--counterexample-backend",
        choices=("graph", "auto", "spot"),
        default=os.environ.get("COUNTEREXAMPLE_BACKEND", "graph"),
        help="'graph' needs no Spot and is exact for the configured formula.",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Load and evaluate a checkpoint without refining or overwriting it.",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Build/load and save the model without running CEGAR or recall.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        help="JSON result path (default: checkpoint name with .summary.json).",
    )
    args = parser.parse_args(argv)

    if args.time_limit_sec < 0:
        parser.error("--time-limit-sec must be nonnegative")
    if args.safety_margin_sec < 0:
        parser.error("--safety-margin-sec must be nonnegative")
    if args.checkpoint_interval_sec < 0:
        parser.error("--checkpoint-interval-sec must be nonnegative")
    if args.max_total_states is not None and args.max_total_states <= 0:
        parser.error("--max-total-states must be positive")
    if args.evaluate_only and args.fresh:
        parser.error("--evaluate-only cannot be combined with --fresh")
    if args.evaluate_only and args.build_only:
        parser.error("--evaluate-only cannot be combined with --build-only")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    grid_size = args.grid_size
    max_total_states = (
        args.max_total_states
        if args.max_total_states is not None
        else (grid_size + 10) ** 3
    )
    checkpoint_path = (
        args.checkpoint.resolve()
        if args.checkpoint is not None
        else _default_checkpoint(grid_size)
    )
    gt_cache_path = (
        args.gt_cache.resolve()
        if args.gt_cache is not None
        else _default_gt_cache(grid_size, args.gt_max_steps)
    )
    summary_path = (
        args.summary.resolve()
        if args.summary is not None
        else _default_summary(checkpoint_path)
    )

    if args.evaluate_only:
        source_path = (
            args.resume_from.resolve()
            if args.resume_from is not None
            else checkpoint_path
        )
        absys, domain, metadata = load_model_checkpoint(source_path)
        actual_shape = abstraction_grid_shape(absys)
        expected_shape = (grid_size, grid_size, grid_size)
        if actual_shape != expected_shape:
            raise ValueError(
                f"Checkpoint grid is {actual_shape}, expected {expected_shape}."
            )
        classification, evaluation = evaluate(
            absys,
            domain,
            grid_size=grid_size,
            gt_cache_path=gt_cache_path,
            gt_max_steps=args.gt_max_steps,
        )
        summary = {
            "stage": "evaluation_only",
            "checkpoint": str(source_path),
            "grid_shape": list(actual_shape),
            "source_metadata": metadata,
            "evaluation": evaluation,
            "evaluated_at": time.time(),
        }
        write_summary(summary_path, summary)
        print(
            f"[RESULT] final recall = "
            f"{evaluation['recall_gt_safe_volume']:.6f}",
            flush=True,
        )
        return 0

    loaded_from: Optional[Path] = None
    prior_metadata: dict = {}
    build_elapsed = 0.0
    if not args.fresh and checkpoint_path.exists():
        loaded_from = checkpoint_path
    elif not args.fresh and args.resume_from is not None:
        loaded_from = args.resume_from.resolve()

    if loaded_from is not None:
        absys, domain, prior_metadata = load_model_checkpoint(loaded_from)
    else:
        print(
            f"[BUILD] constructing {grid_size}x{grid_size}x{grid_size} "
            "partition and transitions...",
            flush=True,
        )
        build_start = time.monotonic()
        absys, domain = build_abstraction(
            grid_size, grid_size, grid_size
        )
        build_elapsed = time.monotonic() - build_start
        print(f"[BUILD] completed in {build_elapsed:.1f}s", flush=True)

    actual_shape = abstraction_grid_shape(absys)
    expected_shape = (grid_size, grid_size, grid_size)
    if actual_shape != expected_shape:
        raise ValueError(
            f"Loaded abstraction grid is {actual_shape}, expected "
            f"{expected_shape}."
        )
    initial_transition_stats = transition_stats(absys)
    completed_candidate_uids = {
        int(uid)
        for uid in prior_metadata.get("completed_candidate_uids", ())
    }

    base_metadata = {
        "grid_shape": list(actual_shape),
        "phi": PHI,
        "loaded_from": str(loaded_from) if loaded_from is not None else None,
        "previous_stage": prior_metadata.get("stage"),
        "build_elapsed_sec": build_elapsed,
        "ordering": args.ordering,
        "rand_seed": args.rand_seed,
        "split_mode": args.split_mode,
        "max_iters_per_cell": args.max_iters_per_cell,
        "max_refine_depth": args.max_refine_depth,
        "min_cell_size": args.min_cell_size,
        "min_cell_theta": args.min_cell_theta,
        "counterexample_backend": args.counterexample_backend,
        "time_limit_sec": args.time_limit_sec,
        "safety_margin_sec": args.safety_margin_sec,
        "max_total_states": max_total_states,
        "gt_cache": str(gt_cache_path),
        "gt_max_steps": args.gt_max_steps,
        "resumed_completed_candidate_count": len(completed_candidate_uids),
        "completed_candidate_uids": completed_candidate_uids,
    }

    # Save a newly constructed model immediately.  This checkpoint already
    # contains the full initial cell partition and transition relation.
    if loaded_from != checkpoint_path or args.fresh:
        save_model_checkpoint(
            absys,
            checkpoint_path,
            domain=domain,
            metadata={
                **base_metadata,
                "stage": "initial_abstraction",
                **initial_transition_stats,
            },
        )

    if args.build_only:
        write_summary(
            summary_path,
            {
                "stage": "build_only",
                "checkpoint": str(checkpoint_path),
                **base_metadata,
                **initial_transition_stats,
            },
        )
        return 0

    cls_before = classify_all_leaves_once(absys, PHI, action="step")
    print("\n[CLASSIFICATION before timed refinement]", flush=True)
    print(f"  verified: {len(cls_before.verified)}", flush=True)
    print(f"  refuted:  {len(cls_before.refuted)}", flush=True)
    print(f"  unknown:  {len(cls_before.unknown)}", flush=True)

    refinement_budget = max(
        0.0, args.time_limit_sec - args.safety_margin_sec
    )
    refinement_start = time.monotonic()
    deadline = refinement_start + refinement_budget
    last_checkpoint = [refinement_start]
    last_progress = [{
        "completed_candidate_uids": completed_candidate_uids,
        "completed_candidate_count": len(completed_candidate_uids),
    }]

    def should_stop() -> bool:
        return (
            STOP_REQUESTED
            or time.monotonic() >= deadline
            or len(absys.part.leaves) >= max_total_states
        )

    def current_stop_reason() -> Optional[str]:
        if len(absys.part.leaves) >= max_total_states:
            return "state_limit"
        if STOP_REQUESTED:
            return "signal"
        if time.monotonic() >= deadline:
            return "time_limit"
        return None

    def progress_callback(absys_cb: Abstraction, progress: dict) -> None:
        last_progress[0] = dict(progress)
        now = time.monotonic()
        checkpoint_due = (
            args.checkpoint_interval_sec == 0
            or now - last_checkpoint[0] >= args.checkpoint_interval_sec
            or should_stop()
        )
        if not checkpoint_due:
            return
        save_model_checkpoint(
            absys_cb,
            checkpoint_path,
            domain=domain,
            metadata={
                **base_metadata,
                **progress,
                "stage": "periodic_refinement",
                "refinement_elapsed_sec": now - refinement_start,
                "stop_reason": current_stop_reason(),
            },
        )
        last_checkpoint[0] = time.monotonic()

    print(
        f"[RUN] CEGAR budget={refinement_budget:.1f}s "
        f"(requested={args.time_limit_sec:.1f}s, "
        f"safety_margin={args.safety_margin_sec:.1f}s), "
        f"state_limit={max_total_states}",
        flush=True,
    )
    try:
        cls_after = refine_one_round(
            absys=absys,
            phi=PHI,
            initial_unknown=set(cls_before.unknown),
            max_iters_per_cell=args.max_iters_per_cell,
            min_cell_width=args.min_cell_size,
            min_cell_height=args.min_cell_size,
            max_refine_depth=args.max_refine_depth,
            min_cell_theta=args.min_cell_theta,
            split_mode=args.split_mode,
            gc_every=args.gc_every,
            ordering=args.ordering,
            rand_seed=args.rand_seed,
            deadline=deadline,
            stop_requested=should_stop,
            progress_callback=progress_callback,
            counterexample_backend=args.counterexample_backend,
            completed_candidate_uids=completed_candidate_uids,
            max_total_states=max_total_states,
        )
    except BaseException:
        save_model_checkpoint(
            absys,
            checkpoint_path,
            domain=domain,
            metadata={
                **base_metadata,
                **last_progress[0],
                "stage": "exception_checkpoint",
                "refinement_elapsed_sec": (
                    time.monotonic() - refinement_start
                ),
            },
        )
        raise

    refinement_elapsed = time.monotonic() - refinement_start
    stopped = (
        should_stop()
        or bool(last_progress[0].get("stopped", False))
    )
    stop_reason = (
        current_stop_reason()
        or last_progress[0].get("stop_reason")
    )

    # Save before evaluation as well as after it.  If evaluation is interrupted,
    # the final refined partition and transitions are still recoverable.
    save_model_checkpoint(
        absys,
        checkpoint_path,
        domain=domain,
        classification=cls_after,
        metadata={
            **base_metadata,
            **last_progress[0],
            "stage": "refinement_complete_pending_evaluation",
            "refinement_elapsed_sec": refinement_elapsed,
            "stopped_by_limit_or_signal": stopped,
            "stop_reason": stop_reason,
        },
    )

    cls_after, evaluation = evaluate(
        absys,
        domain,
        grid_size=grid_size,
        gt_cache_path=gt_cache_path,
        gt_max_steps=args.gt_max_steps,
        classification=cls_after,
    )
    final_metadata = {
        **base_metadata,
        **last_progress[0],
        "stage": "final",
        "refinement_elapsed_sec": refinement_elapsed,
        "stopped_by_limit_or_signal": stopped,
        "stop_reason": stop_reason,
        "evaluation": evaluation,
    }
    save_model_checkpoint(
        absys,
        checkpoint_path,
        domain=domain,
        classification=cls_after,
        metadata=final_metadata,
    )

    summary = {
        "stage": "final",
        "checkpoint": str(checkpoint_path),
        **final_metadata,
    }
    write_summary(summary_path, summary)
    print("\n[RESULT]", flush=True)
    print(f"  checkpoint: {checkpoint_path}", flush=True)
    print(f"  summary:    {summary_path}", flush=True)
    print(f"  leaves:     {len(absys.part.leaves)}", flush=True)
    print(
        f"  recall:     {evaluation['recall_gt_safe_volume']:.6f} "
        f"({evaluation['recall_gt_safe_volume'] * 100:.2f}%)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
