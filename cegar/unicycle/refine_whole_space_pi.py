#!/usr/bin/env python3
from __future__ import annotations

import os
import pickle
import signal
import sys
import time
from pathlib import Path
from typing import Any, Tuple

from abstraction import Rect
from gt_cache import default_gt_cache_path, load_gt_cache
from refine_whole_space import (
    classify_all_leaves_once,
    compute_metrics,
    refine_one_round,
    save_model_checkpoint,
)

STOP_REQUESTED = False


def _request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"\n[STOP] received signal {signum}; will checkpoint and exit cleanly.", flush=True)


for _sig_name in ("SIGTERM", "SIGINT", "SIGUSR1"):
    _sig = getattr(signal, _sig_name, None)
    if _sig is not None:
        try:
            signal.signal(_sig, _request_stop)
        except Exception:
            pass


def _deadline_from_env(default_seconds: int = 3 * 60 * 60, safety_margin_sec: int = 300) -> float:
    raw = os.environ.get("TIME_LIMIT_SEC", str(default_seconds))
    try:
        limit = int(raw)
    except ValueError:
        limit = default_seconds
    try:
        margin = int(os.environ.get("TIME_LIMIT_SAFETY_MARGIN_SEC", str(safety_margin_sec)))
    except ValueError:
        margin = safety_margin_sec
    limit = max(0, limit - max(0, margin))
    return time.monotonic() + limit


def _should_stop(deadline: float | None = None) -> bool:
    return STOP_REQUESTED or (deadline is not None and time.monotonic() >= deadline)


def load_model_checkpoint(path: str | Path) -> Tuple[Any, Rect, dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing model checkpoint {path}. Run the build/checkpoint job first."
        )

    with path.open("rb") as f:
        payload = pickle.load(f)

    if not isinstance(payload, dict) or "absys" not in payload:
        raise ValueError(
            f"Checkpoint {path} does not look like a model checkpoint created by save_model_checkpoint()."
        )

    absys = payload["absys"]
    domain = payload.get("domain")
    metadata = payload.get("metadata", {})
    if domain is None:
        from main import X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX
        domain = Rect(X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX)

    if not isinstance(metadata, dict):
        metadata = {"raw_metadata": metadata}

    print(f"[LOAD] checkpoint: {path}", flush=True)
    print(f"[LOAD] metadata: {metadata}", flush=True)
    return absys, domain, metadata


def main() -> int:
    checkpoint_path = Path(os.environ.get(
        "MODEL_CHECKPOINT",
        "artifacts/unicycle_refinement_model.pkl",
    ))
    deadline = _deadline_from_env()

    ordering = os.environ.get("ORDERING", "largest")
    rand_seed = int(os.environ.get("RAND_SEED", "0"))
    split_mode = os.environ.get("SPLIT_MODE", "auto")
    max_iters_per_cell = int(os.environ.get("MAX_ITERS_PER_CELL", "150"))
    max_refine_depth = int(os.environ.get("MAX_REFINE_DEPTH", "40"))
    min_cell_size = float(os.environ.get("MIN_CELL_SIZE", "0.001"))
    min_cell_theta = float(os.environ.get("MIN_CELL_THETA", "0.001"))
    gc_every = int(os.environ.get("GC_EVERY", "100"))

    print(f"[WRAPPER] loading abstraction from checkpoint {checkpoint_path}", flush=True)
    absys, domain, metadata = load_model_checkpoint(checkpoint_path)

    # Load the prebuilt ground-truth cache for counterexample validation.
    gt_nx = int(os.environ.get("GT_NX", "90"))
    gt_ny = int(os.environ.get("GT_NY", "90"))
    gt_nz = int(os.environ.get("GT_NZ", "90"))
    gt_max_steps = int(os.environ.get("GT_MAX_STEPS", "10"))
    gt_cache_path = Path(os.environ.get(
        "GT_CACHE_PATH",
        str(default_gt_cache_path(gt_nx, gt_ny, gt_nz, gt_max_steps)),
    ))
    print(f"[WRAPPER] loading GT cache from {gt_cache_path}", flush=True)
    gt_payload = load_gt_cache(gt_cache_path)
    if (
        int(gt_payload["nx_gt"]) != gt_nx or
        int(gt_payload["ny_gt"]) != gt_ny or
        int(gt_payload["nz_gt"]) != gt_nz or
        int(gt_payload["max_steps"]) != gt_max_steps
    ):
        raise ValueError(
            f"GT cache {gt_cache_path} does not match requested grid/steps "
            f"{gt_nx}x{gt_ny}x{gt_nz}, steps={gt_max_steps}."
        )

    phi = os.environ.get("PHI", "(!unsafe) U goal")
    print(f"[WRAPPER] phi: {phi}", flush=True)

    cls_before = classify_all_leaves_once(absys, phi, action="step")
    verified_before = set(cls_before.verified)
    refuted_before = set(cls_before.refuted)
    unknown_before = set(cls_before.unknown)

    print("\n[CLASSIFICATION Before Refinement]")
    print("verified:", len(verified_before))
    print("refuted :", len(refuted_before))
    print("unknown :", len(unknown_before))
    compute_metrics(absys, verified_before, refuted_before, unknown_before)

    if _should_stop(deadline):
        save_model_checkpoint(
            absys,
            checkpoint_path,
            domain=domain,
            metadata={
                "stage": "pre_refinement_stop",
                "reason": "deadline_or_signal",
                "loaded_metadata": metadata,
            },
        )
        return 0

    cls_after = refine_one_round(
        absys=absys,
        phi=phi,
        initial_unknown=unknown_before,
        max_iters_per_cell=max_iters_per_cell,
        min_cell_width=min_cell_size,
        min_cell_height=min_cell_size,
        max_refine_depth=max_refine_depth,
        min_cell_theta=min_cell_theta,
        split_mode=split_mode,
        gc_every=gc_every,
        ordering=ordering,
        rand_seed=rand_seed,
        deadline=deadline,
        checkpoint_path=checkpoint_path,
        gt_cache_payload=gt_payload,
    )

    save_model_checkpoint(
        absys,
        checkpoint_path,
        domain=domain,
        metadata={
            "stage": "post_refinement",
            "ordering": ordering,
            "rand_seed": rand_seed,
            "split_mode": split_mode,
            "max_iters_per_cell": max_iters_per_cell,
            "max_refine_depth": max_refine_depth,
            "min_cell_theta": min_cell_theta,
            "loaded_metadata": metadata,
        },
    )

    verified_after = set(cls_after.verified)
    refuted_after = set(cls_after.refuted)
    unknown_after = set(cls_after.unknown)

    print("\n[CLASSIFICATION After Refinement]")
    print("verified:", len(verified_after))
    print("refuted :", len(refuted_after))
    print("unknown :", len(unknown_after))
    compute_metrics(absys, verified_after, refuted_after, unknown_after)

    if _should_stop(deadline):
        print("[WRAPPER] stopped at deadline; checkpoint saved.", flush=True)
        return 0

    print("\n── Summary ───────────────────────────────────────────────────────")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Ordering:   {ordering!r}" + (f"  seed={rand_seed}" if ordering == "random" else ""))
    print(f"  Split mode: {split_mode!r}")
    print(f"  Leaves:     {len(absys.part.leaves)}")
    print(f"──────────────────────────────────────────────────────────────────")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())