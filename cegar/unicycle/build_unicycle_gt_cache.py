from __future__ import annotations

import argparse
import os

from gt_cache import default_gt_cache_path, load_or_build_gt_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and save the unicycle GT safe set cache.")
    parser.add_argument("--nx-gt", type=int, default=int(os.environ.get("GT_NX", "90")))
    parser.add_argument("--ny-gt", type=int, default=int(os.environ.get("GT_NY", "90")))
    parser.add_argument("--nz-gt", type=int, default=int(os.environ.get("GT_NZ", "90")))
    parser.add_argument("--max-steps", type=int, default=int(os.environ.get("GT_MAX_STEPS", "10")))
    parser.add_argument("--cache-path", type=str, default=os.environ.get("GT_CACHE_PATH"))
    parser.add_argument("--force", action="store_true", default=os.environ.get("FORCE_REBUILD_GT", "0") == "1")
    args = parser.parse_args()

    cache_path = args.cache_path or default_gt_cache_path(args.nx_gt, args.ny_gt, args.nz_gt, args.max_steps)
    print(f"[GT] cache path: {cache_path}")
    load_or_build_gt_cache(
        cache_path,
        nx_gt=args.nx_gt,
        ny_gt=args.ny_gt,
        nz_gt=args.nz_gt,
        max_steps=args.max_steps,
        force_rebuild=args.force,
    )


if __name__ == "__main__":
    main()