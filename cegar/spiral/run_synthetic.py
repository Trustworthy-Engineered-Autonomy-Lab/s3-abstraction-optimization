#!/usr/bin/env python3
"""Timed, resumable CEGAR runner for the 2-D synthetic benchmark."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import signal
import sys
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")

from abstraction import (
    Abstraction,
    Rect,
    compute_verified_set_via_fixpoint,
)
from cegar_loop import CEGARResult, run_cegar
from synthetic_benchmark import (
    DOMAIN,
    FORMULA,
    build_abstraction,
    compute_synthetic_v3_recall,
)


SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = SCRIPT_DIR / "artifacts"
SYNTHETIC_V3_GT = (
    SCRIPT_DIR.parents[1]
    / "synthetic-v3"
    / "synthetic_reach_regions.pkl"
)
CHECKPOINT_VERSION = 2
CEGAR_SEMANTICS = "synthetic-v3-cell-oracle-v1"
STOP_REQUESTED = False


def _request_stop(signum, frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(
        f"\n[STOP] received signal {signum}; stopping at the next safe "
        "CEGAR iteration boundary.",
        flush=True,
    )


for _signal_name in ("SIGTERM", "SIGINT", "SIGUSR1"):
    _signal = getattr(signal, _signal_name, None)
    if _signal is not None:
        try:
            signal.signal(_signal, _request_stop)
        except (OSError, RuntimeError, ValueError):
            pass


def default_checkpoint_path(grid_size: int) -> Path:
    return ARTIFACT_DIR / f"synthetic_cegar_{grid_size}x{grid_size}.pkl"


def default_gt_reach_regions_path() -> Path:
    return SYNTHETIC_V3_GT


def default_summary_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_suffix(".summary.json")


def save_model_checkpoint(
    absys: Abstraction,
    path: str | Path,
    *,
    domain: Rect = DOMAIN,
    metadata: Optional[dict] = None,
    verified: Optional[set[int]] = None,
) -> Path:
    """Atomically save cells, cell IDs, transitions, and run metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "absys": absys,
        "domain": domain,
        "metadata": dict(metadata or {}),
        "saved_wall_time": time.time(),
        "saved_monotonic": time.monotonic(),
    }
    if verified is not None:
        payload["classification"] = {
            "verified": set(verified),
            "unknown": set(absys.part.leaves) - set(verified),
            "refuted": set(),
        }

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)
    print(
        f"[CHECKPOINT] saved {len(absys.part.leaves)} leaves and "
        f"{len(absys.tr.succ)} transition sources to {path}",
        flush=True,
    )
    return path


def load_model_checkpoint(
    path: str | Path,
) -> Tuple[Abstraction, Rect, dict]:
    """Load either a new dict checkpoint or the legacy raw Abstraction."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint does not exist: {path}")
    with path.open("rb") as stream:
        payload = pickle.load(stream)

    if isinstance(payload, Abstraction):
        absys = payload
        domain = DOMAIN
        metadata = {
            "stage": "legacy_raw_abstraction",
            "legacy_path": str(path),
        }
    elif isinstance(payload, dict) and "absys" in payload:
        absys = payload["absys"]
        domain = payload.get("domain") or DOMAIN
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {"raw_metadata": repr(metadata)}
    else:
        raise ValueError(
            f"{path} is neither a raw synthetic Abstraction nor a checkpoint."
        )

    print(f"[LOAD] checkpoint: {path}", flush=True)
    print(
        f"[LOAD] leaves={len(absys.part.leaves)} "
        f"transition_sources={len(absys.tr.succ)} "
        f"stage={metadata.get('stage')!r}",
        flush=True,
    )
    return absys, domain, metadata


def abstraction_grid_shape(absys: Abstraction) -> Tuple[int, int]:
    roots = absys.part.roots
    return (
        len({(node.rect.xmin, node.rect.xmax) for node in roots}),
        len({(node.rect.ymin, node.rect.ymax) for node in roots}),
    )


def transition_stats(absys: Abstraction, *, validate: bool = True) -> dict:
    valid_uids = set(absys.part.leaves)
    valid_uids.add(absys.OUT_UID)
    missing = [uid for uid in absys.part.leaves if uid not in absys.tr.succ]
    dangling = 0
    edge_count = 0
    for by_action in absys.tr.succ.values():
        for destinations in by_action.values():
            edge_count += len(destinations)
            if validate:
                dangling += sum(dst not in valid_uids for dst in destinations)
    if validate and (missing or dangling):
        raise ValueError(
            "Incomplete transition relation: "
            f"missing_sources={len(missing)}, dangling_edges={dangling}."
        )
    return {
        "leaf_count": len(absys.part.leaves),
        "transition_source_count": len(absys.tr.succ),
        "transition_edge_count": edge_count,
        "missing_transition_sources": len(missing),
        "dangling_transition_edges": dangling,
    }


def ensure_current_transition_semantics(
    absys: Abstraction,
    metadata: Optional[dict] = None,
) -> bool:
    """Rebuild legacy transition graphs using synthetic-v3 semantics."""
    current = getattr(absys, "TRANSITION_SEMANTICS", None)
    recorded = (metadata or {}).get("transition_semantics")
    if recorded == current:
        return False
    print(
        "[TRANSITIONS] rebuilding with synthetic-v3 AABB and mixed-OUT "
        "semantics...",
        flush=True,
    )
    absys.rebuild_all_transitions()
    return True


def load_gt_reach_regions(path: Path) -> dict:
    """Load synthetic-v3's fixed 100x100 goal/fail reference."""
    if not path.exists():
        raise FileNotFoundError(
            f"synthetic-v3 ground truth does not exist: {path}"
        )
    with path.open("rb") as stream:
        regions = pickle.load(stream)
    if (
        not isinstance(regions, dict)
        or not regions
        or not all(
            isinstance(key, tuple)
            and len(key) == 2
            and label in {"goal", "fail", "unk"}
            for key, label in regions.items()
        )
    ):
        raise ValueError(
            f"{path} is not a synthetic-v3 reach-region dictionary."
        )
    counts = {}
    for label in regions.values():
        counts[label] = counts.get(label, 0) + 1
    print(
        f"[GT] loaded {len(regions)} synthetic-v3 cells from {path}: "
        f"{counts}",
        flush=True,
    )
    return regions


def evaluate(
    absys: Abstraction,
    *,
    gt_reach_regions_path: Path,
) -> Tuple[set[int], dict]:
    start = time.monotonic()
    verified = compute_verified_set_via_fixpoint(absys)
    verification_elapsed = time.monotonic() - start
    print(
        f"[EVAL] verified={len(verified)}/{len(absys.part.leaves)} "
        f"in {verification_elapsed:.3f}s",
        flush=True,
    )

    gt_reach_regions = load_gt_reach_regions(gt_reach_regions_path)
    recall_details = compute_synthetic_v3_recall(
        absys,
        verified,
        gt_reach_regions,
        initial_domain=DOMAIN,
    )
    result = {
        "verified_cells": len(verified),
        "unknown_cells": len(absys.part.leaves) - len(verified),
        "verification_elapsed_sec": verification_elapsed,
        "recall_definition": "synthetic-v3-volume",
        "recall_gt_safe_volume": recall_details["recall"],
        "gt_reach_regions": str(gt_reach_regions_path),
        **{
            key: value
            for key, value in recall_details.items()
            if key != "recall"
        },
        **transition_stats(absys),
    }
    return verified, result


def refine_unknown_cells(
    absys: Abstraction,
    unknown_uids: set[int],
    *,
    max_iters_per_cell: int,
    max_refine_depth: int,
    min_cell_size: float,
    split_mode: str,
    counterexample_backend: str,
    max_total_states: int,
    stop_requested: Callable[[], bool],
    progress_callback: Callable[[Abstraction, dict], None],
    completed_candidate_uids: set[int],
) -> dict:
    """Mirror the unicycle workflow: one bounded CEGAR run per unknown cell."""
    def area(uid: int) -> float:
        rect = absys.part.leaves[uid].rect
        return float(
            (rect.xmax - rect.xmin) * (rect.ymax - rect.ymin)
        )

    candidates = [
        uid
        for uid in unknown_uids
        if uid in absys.part.leaves
        and uid not in completed_candidate_uids
    ]
    candidates.sort(key=lambda uid: (-area(uid), uid))

    total_refinements = 0
    total_iterations = 0
    total_ignored = 0
    verified_runs = 0
    unrefinable_runs = 0
    processed = 0
    stop_reason = None

    print(
        f"[REFINE] unknown candidates={len(candidates)}, "
        f"already_completed={len(completed_candidate_uids)}",
        flush=True,
    )
    for index, uid in enumerate(candidates):
        if len(absys.part.leaves) >= max_total_states:
            stop_reason = "state_limit"
            break
        if stop_requested():
            stop_reason = "external_stop"
            break
        if index % 200 == 0:
            print(
                f"  [refine] {index}/{len(candidates)} "
                f"leaves={len(absys.part.leaves)} "
                f"splits={total_refinements}",
                flush=True,
            )

        if uid not in absys.part.leaves:
            completed_candidate_uids.add(uid)
            processed += 1
            continue

        result = run_cegar(
            absys,
            {uid},
            phi=FORMULA,
            action="step",
            max_iters=max_iters_per_cell,
            min_cell_width=min_cell_size,
            min_cell_height=min_cell_size,
            max_refine_depth=max_refine_depth,
            split_mode=split_mode,
            verbose=False,
            counterexample_backend=counterexample_backend,
            stop_requested=stop_requested,
            max_total_states=max_total_states,
        )
        processed += 1
        total_refinements += result.refinements
        total_iterations += result.iterations
        total_ignored += result.ignored_counterexamples
        if result.verified:
            verified_runs += 1
        if result.stop_reason == "unrefinable_counterexample":
            unrefinable_runs += 1

        resource_stop = result.stop_reason in {
            "state_limit",
            "external_stop",
        }
        if not resource_stop:
            completed_candidate_uids.add(uid)

        progress = {
            "processed": processed,
            "ordered_total": len(candidates),
            "refinements": total_refinements,
            "iterations": total_iterations,
            "ignored_counterexamples": total_ignored,
            "verified_cell_runs": verified_runs,
            "unrefinable_cell_runs": unrefinable_runs,
            "completed_candidate_uids": completed_candidate_uids,
            "completed_candidate_count": len(completed_candidate_uids),
            "current_total_states": len(absys.part.leaves),
            "max_total_states": max_total_states,
            "stop_reason": result.stop_reason if resource_stop else None,
        }
        progress_callback(absys, progress)

        if resource_stop:
            stop_reason = result.stop_reason
            break

    if stop_reason is None:
        if len(absys.part.leaves) >= max_total_states:
            stop_reason = "state_limit"
        elif stop_requested():
            stop_reason = "external_stop"
        else:
            stop_reason = "completed_pass"

    summary = {
        "processed": processed,
        "ordered_total": len(candidates),
        "refinements": total_refinements,
        "iterations": total_iterations,
        "ignored_counterexamples": total_ignored,
        "verified_cell_runs": verified_runs,
        "unrefinable_cell_runs": unrefinable_runs,
        "completed_candidate_uids": completed_candidate_uids,
        "completed_candidate_count": len(completed_candidate_uids),
        "current_total_states": len(absys.part.leaves),
        "max_total_states": max_total_states,
        "stop_reason": stop_reason,
    }
    progress_callback(absys, summary)
    print(
        f"[REFINE] done: processed={processed}/{len(candidates)}, "
        f"splits={total_refinements}, leaves={len(absys.part.leaves)}, "
        f"stop={stop_reason}",
        flush=True,
    )
    return summary


def write_summary(path: Path, summary: dict) -> None:
    def json_default(value):
        if isinstance(value, set):
            return sorted(value)
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, CEGARResult):
            return value.__dict__
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
            "Build/resume the synthetic abstraction, run bounded CEGAR, "
            "save the complete model, and compute final recall."
        )
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        choices=(60, 90),
        required=True,
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Seed a missing output checkpoint from this new or legacy pickle.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore existing checkpoints and build a fresh uniform model.",
    )
    parser.add_argument(
        "--time-limit-sec",
        type=float,
        default=float(os.environ.get("TIME_LIMIT_SEC", 3 * 60 * 60)),
    )
    parser.add_argument(
        "--safety-margin-sec",
        type=float,
        default=float(os.environ.get("TIME_LIMIT_SAFETY_MARGIN_SEC", 0)),
    )
    parser.add_argument(
        "--max-total-states",
        type=int,
        help="Live-cell cap; default is (grid_size + 10)^2.",
    )
    parser.add_argument(
        "--checkpoint-interval-sec",
        type=float,
        default=float(os.environ.get("CHECKPOINT_INTERVAL_SEC", 15 * 60)),
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
        default=float(os.environ.get("MIN_CELL_SIZE", 1e-6)),
    )
    parser.add_argument(
        "--split-mode",
        choices=("auto", "xy"),
        default=os.environ.get("SPLIT_MODE", "auto"),
    )
    parser.add_argument(
        "--counterexample-backend",
        choices=("graph", "auto", "spot"),
        default=os.environ.get("COUNTEREXAMPLE_BACKEND", "graph"),
    )
    parser.add_argument(
        "--gt-reach-regions",
        "--gt-cache",
        dest="gt_reach_regions",
        type=Path,
        help=(
            "synthetic-v3 reach-region pickle. --gt-cache is retained as "
            "a compatibility alias."
        ),
    )
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args(argv)

    if args.time_limit_sec < 0:
        parser.error("--time-limit-sec must be nonnegative")
    if args.safety_margin_sec < 0:
        parser.error("--safety-margin-sec must be nonnegative")
    if args.checkpoint_interval_sec < 0:
        parser.error("--checkpoint-interval-sec must be nonnegative")
    if args.max_total_states is not None and args.max_total_states <= 0:
        parser.error("--max-total-states must be positive")
    if args.max_iters_per_cell <= 0:
        parser.error("--max-iters-per-cell must be positive")
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
        else (grid_size + 10) ** 2
    )
    checkpoint_path = (
        args.checkpoint.resolve()
        if args.checkpoint is not None
        else default_checkpoint_path(grid_size)
    )
    gt_reach_regions_path = (
        args.gt_reach_regions.resolve()
        if args.gt_reach_regions is not None
        else default_gt_reach_regions_path()
    )
    summary_path = (
        args.summary.resolve()
        if args.summary is not None
        else default_summary_path(checkpoint_path)
    )

    if args.evaluate_only:
        source_path = (
            args.resume_from.resolve()
            if args.resume_from is not None
            else checkpoint_path
        )
        absys, _, metadata = load_model_checkpoint(source_path)
        ensure_current_transition_semantics(absys, metadata)
        expected_shape = (grid_size, grid_size)
        actual_shape = abstraction_grid_shape(absys)
        if actual_shape != expected_shape:
            raise ValueError(
                f"Checkpoint root grid is {actual_shape}, "
                f"expected {expected_shape}."
            )
        _, evaluation = evaluate(
            absys,
            gt_reach_regions_path=gt_reach_regions_path,
        )
        write_summary(
            summary_path,
            {
                "stage": "evaluation_only",
                "checkpoint": str(source_path),
                "grid_shape": list(actual_shape),
                "source_metadata": metadata,
                "evaluation": evaluation,
                "evaluated_at": time.time(),
            },
        )
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
        transitions_rebuilt = ensure_current_transition_semantics(
            absys,
            prior_metadata,
        )
    else:
        print(
            f"[BUILD] constructing {grid_size}x{grid_size} abstraction...",
            flush=True,
        )
        build_start = time.monotonic()
        absys, domain = build_abstraction(grid_size, grid_size)
        build_elapsed = time.monotonic() - build_start
        transitions_rebuilt = False
        print(f"[BUILD] completed in {build_elapsed:.3f}s", flush=True)

    actual_shape = abstraction_grid_shape(absys)
    expected_shape = (grid_size, grid_size)
    if actual_shape != expected_shape:
        raise ValueError(
            f"Loaded root grid is {actual_shape}, expected {expected_shape}."
        )
    initial_transition_stats = transition_stats(absys)
    resume_semantics_match = (
        prior_metadata.get("transition_semantics")
        == absys.TRANSITION_SEMANTICS
        and prior_metadata.get("cegar_semantics") == CEGAR_SEMANTICS
    )
    completed_candidate_uids = (
        {
            int(uid)
            for uid in prior_metadata.get("completed_candidate_uids", ())
        }
        if resume_semantics_match
        else set()
    )
    if loaded_from is not None and not resume_semantics_match:
        print(
            "[RESUME] prior completed-candidate markers were cleared because "
            "the transition/CEGAR semantics changed.",
            flush=True,
        )
    base_metadata = {
        "grid_shape": list(actual_shape),
        "phi": FORMULA,
        "loaded_from": str(loaded_from) if loaded_from is not None else None,
        "previous_stage": prior_metadata.get("stage"),
        "build_elapsed_sec": build_elapsed,
        "time_limit_sec": args.time_limit_sec,
        "safety_margin_sec": args.safety_margin_sec,
        "max_total_states": max_total_states,
        "max_iters_per_cell": args.max_iters_per_cell,
        "max_refine_depth": args.max_refine_depth,
        "min_cell_size": args.min_cell_size,
        "split_mode": args.split_mode,
        "counterexample_backend": args.counterexample_backend,
        "transition_semantics": absys.TRANSITION_SEMANTICS,
        "cegar_semantics": CEGAR_SEMANTICS,
        "transitions_rebuilt_on_load": transitions_rebuilt,
        "gt_reach_regions": str(gt_reach_regions_path),
        "completed_candidate_uids": completed_candidate_uids,
        "resumed_completed_candidate_count": len(
            completed_candidate_uids
        ),
    }

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

    verified_before = compute_verified_set_via_fixpoint(absys)
    unknown_before = set(absys.part.leaves) - verified_before
    print(
        f"[CLASSIFY] before refinement: verified={len(verified_before)} "
        f"unknown={len(unknown_before)}",
        flush=True,
    )

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

    def current_stop_reason() -> Optional[str]:
        if len(absys.part.leaves) >= max_total_states:
            return "state_limit"
        if STOP_REQUESTED:
            return "signal"
        if time.monotonic() >= deadline:
            return "time_limit"
        return None

    def should_stop() -> bool:
        return current_stop_reason() is not None

    def progress_callback(absys_cb: Abstraction, progress: dict) -> None:
        last_progress[0] = dict(progress)
        now = time.monotonic()
        checkpoint_due = (
            args.checkpoint_interval_sec == 0
            or now - last_checkpoint[0] >= args.checkpoint_interval_sec
            or should_stop()
            or progress.get("stop_reason") is not None
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
                "stop_reason": (
                    current_stop_reason()
                    or progress.get("stop_reason")
                ),
            },
        )
        last_checkpoint[0] = time.monotonic()

    print(
        f"[RUN] CEGAR budget={refinement_budget:.1f}s, "
        f"state_limit={max_total_states}, "
        f"unknown_initial_states={len(unknown_before)}",
        flush=True,
    )
    try:
        refinement_summary = refine_unknown_cells(
            absys,
            unknown_before,
            max_iters_per_cell=args.max_iters_per_cell,
            max_refine_depth=args.max_refine_depth,
            min_cell_size=args.min_cell_size,
            split_mode=args.split_mode,
            counterexample_backend=args.counterexample_backend,
            max_total_states=max_total_states,
            stop_requested=should_stop,
            progress_callback=progress_callback,
            completed_candidate_uids=completed_candidate_uids,
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

    refinement_finished_at = time.monotonic()
    refinement_elapsed = refinement_finished_at - refinement_start
    stop_reason = (
        current_stop_reason()
        or refinement_summary.get("stop_reason")
        or last_progress[0].get("stop_reason")
    )
    refinement_summary["stop_reason"] = stop_reason
    state_limit_before_time_limit = (
        stop_reason == "state_limit"
        and refinement_finished_at < deadline
    )
    cegar_timer = {
        "cegar_runtime_sec": refinement_elapsed,
        "state_limit_reached_before_time_limit": (
            state_limit_before_time_limit
        ),
        "cegar_runtime_to_state_limit_sec": (
            refinement_elapsed
            if state_limit_before_time_limit
            else None
        ),
    }
    refinement_summary.update(cegar_timer)
    if state_limit_before_time_limit:
        print(
            "[TIMER] state limit reached before time limit; "
            f"total CEGAR runtime={refinement_elapsed:.3f}s",
            flush=True,
        )

    save_model_checkpoint(
        absys,
        checkpoint_path,
        domain=domain,
        metadata={
            **base_metadata,
            **last_progress[0],
            "stage": "refinement_complete_pending_evaluation",
            "refinement_elapsed_sec": refinement_elapsed,
            **cegar_timer,
            "refinement_summary": refinement_summary,
            "stop_reason": stop_reason,
        },
    )

    verified, evaluation = evaluate(
        absys,
        gt_reach_regions_path=gt_reach_regions_path,
    )
    final_metadata = {
        **base_metadata,
        **last_progress[0],
        "stage": "final",
        "refinement_elapsed_sec": refinement_elapsed,
        **cegar_timer,
        "refinement_summary": refinement_summary,
        "stop_reason": stop_reason,
        "evaluation": evaluation,
    }
    save_model_checkpoint(
        absys,
        checkpoint_path,
        domain=domain,
        metadata=final_metadata,
        verified=verified,
    )
    write_summary(
        summary_path,
        {
            "stage": "final",
            "checkpoint": str(checkpoint_path),
            **final_metadata,
        },
    )

    print("\n[RESULT]", flush=True)
    print(f"  checkpoint: {checkpoint_path}", flush=True)
    print(f"  summary:    {summary_path}", flush=True)
    print(f"  leaves:     {len(absys.part.leaves)}", flush=True)
    print(f"  stop:       {stop_reason}", flush=True)
    if state_limit_before_time_limit:
        print(
            f"  CEGAR time: {refinement_elapsed:.3f}s "
            "(state limit reached before time limit)",
            flush=True,
        )
    print(
        f"  recall:     {evaluation['recall_gt_safe_volume']:.6f} "
        f"({evaluation['recall_gt_safe_volume'] * 100:.2f}%)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
