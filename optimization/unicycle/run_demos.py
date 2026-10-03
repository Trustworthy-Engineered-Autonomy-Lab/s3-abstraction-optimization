# =====================================================================
# Description: runs the abstraction and verification demos on a single
# shared abstraction, and tabulates shared and demo-specific results.
# =====================================================================

# =====================================================================
# Libraries
# =====================================================================

import unicycle_abstraction as ua
import verification_tools as vt
import numpy as np
import pandas as pd
import pickle as pkl
import pyModelChecking as pmc
import time
import argparse
from pathlib import Path


def metric_table(rows):
    # Keep ints as ints; a mixed int/float column would coerce everything to float
    return pd.DataFrame({
        'Metric': [r[0] for r in rows],
        'Value': pd.Series([r[1] for r in rows], dtype=object),
    })


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the abstraction and verification demos and "
                    "summarize their results in tables."
    )
    parser.add_argument("--shape", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--rollouts", type=int, default=5)
    parser.add_argument("--x0", type=float, nargs=3, default=[5.0, 10.0, 0.0])
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":

    cli_args = parse_args()

    base_dir = Path(__file__).parent
    out_dir = base_dir / 'demo_artifacts'
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_reach_fname = base_dir / 'artifacts' / 'cache' / 'reach.pkl'

    # Settings shared by both demos
    abstraction_shape = [cli_args.shape] * 3
    domain_lb = np.array([0.0, 0.0, -np.pi])
    domain_ub = np.array([50.0, 50.0, np.pi])
    init_domain_lb = np.array([0.0, 0.0, -np.pi/4])
    init_domain_ub = np.array([50.0, 50.0, np.pi/4])
    goal_center, goal_radius = np.array([40.0, 20.0]), 8.0
    obs_center, obs_radius = np.array([25.0, 25.0]), 5.0

    # Build the abstraction once
    x_edges, y_edges, theta_edges = [
        np.linspace(lb, ub, n + 1)
        for lb, ub, n in zip(domain_lb, domain_ub, abstraction_shape)
    ]
    t0 = time.perf_counter()
    kripke_components = ua.build_abstraction(
        x_edges, y_edges, theta_edges,
        verbose=True,
        goal_center=goal_center, goal_radius=goal_radius,
        obs_center=obs_center, obs_radius=obs_radius,
    )
    build_time = time.perf_counter() - t0
    successors, cells = ua.kripke_to_dicts(
        kripke_components, x_edges, y_edges, theta_edges)

    # Abstraction demo: concrete trajectory vs. abstract reachable sets
    traj = ua.simulate_concrete(cli_args.x0, cli_args.horizon)
    traj_ids = [ua.locate_cell(s, x_edges, y_edges, theta_edges) for s in traj]
    frontiers = ua.abstract_frontiers(successors, traj_ids[0], cli_args.horizon)
    contained = [cid in fr for cid, fr in zip(traj_ids, frontiers)]
    rollouts = ua.sample_abstract_rollouts(
        successors, traj_ids[0], cli_args.horizon,
        cli_args.rollouts, np.random.default_rng(cli_args.seed))
    ua.plot_overlay(
        traj, frontiers, rollouts, cells, x_edges, y_edges,
        goal_center=goal_center, goal_radius=goal_radius,
        obs_center=obs_center, obs_radius=obs_radius,
        save_path=out_dir / 'abstraction_overlay.png')

    # Verification demo: model check A (safe U goal) against ground truth
    _, init_states = ua.init_cells_to_ids(
        init_domain_lb, init_domain_ub, x_edges, y_edges, theta_edges)
    kripke_structure = pmc.Kripke(
        S=kripke_components['kripke_states'],
        S0=init_states,
        R=list(kripke_components['kripke_transitions']),
        L=kripke_components['kripke_labels'])
    t0 = time.perf_counter()
    sat_init_states = vt.model_check_kripke(kripke_structure)
    check_time = time.perf_counter() - t0

    with open(gt_reach_fname, "rb") as f:
        gt_reach_regions = pkl.load(f)
    ground_truth_check = vt.check_ground_truth_fast(
        None, abstraction_shape, domain_lb, domain_ub, gt_reach_regions,
        edges=(x_edges, y_edges, theta_edges))
    init_vol = sum(ground_truth_check[s][1] for s in init_states)
    mc_vol = sum(ground_truth_check[s][1] for s in sat_init_states)
    gt_sat = [s for s in init_states if ground_truth_check[s][0] == "goal"]
    gt_vol = sum(ground_truth_check[s][1] for s in gt_sat)
    not_gt_goal = [s for s in sat_init_states
                   if ground_truth_check[s][0] != "goal"]
    vt.plot_verification_slice(
        sat_init_states, init_states, ground_truth_check,
        x_edges, y_edges, theta_edges, out_dir / 'verification_slice.png')

    # Shared metrics: computed once from the single abstraction
    stats = ua.summarize_abstraction(kripke_components)
    shared = metric_table([
        ('States', stats['states']),
        ('Transitions', stats['transitions']),
        ('Out-degree mean', round(stats['out_degree_mean'], 2)),
        ('Out-degree min', stats['out_degree_min']),
        ('Out-degree max', stats['out_degree_max']),
        ('Cell self-loops', stats['cell_self_loops']),
        ('Edges into OOB state', stats['edges_to_oob']),
        ('Deadlock states', stats['deadlock_states']),
        ('Label: safe', stats['label_counts'].get('safe', 0)),
        ('Label: goal', stats['label_counts'].get('goal', 0)),
        ('Label: fail', stats['label_counts'].get('fail', 0)),
        ('Build time (s)', round(build_time, 2)),
    ])

    # Abstraction-demo-only: reachable states per step
    reachable = pd.DataFrame({
        'Step': range(len(frontiers)),
        'Concrete cell': traj_ids,
        'Abstract states reachable': [len(fr) for fr in frontiers],
        'Concrete cell contained': contained,
    })

    # Verification-demo-only: model checking results
    verification = metric_table([
        ('Initial cells', len(init_states)),
        ('Verified initial cells', len(sat_init_states)),
        ('Ground-truth goal cells', len(gt_sat)),
        ('Verified volume fraction', round(mc_vol / init_vol, 4)),
        ('Recall', round(mc_vol / gt_vol, 4) if gt_vol > 0 else float('nan')),
        ('Verified but not ground-truth goal', len(not_gt_goal)),
        ('Model check time (s)', round(check_time, 2)),
    ])

    shared.to_csv(out_dir / 'shared_metrics.csv', index=False)
    reachable.to_csv(out_dir / 'reachable_states.csv', index=False)
    verification.to_csv(out_dir / 'verification_results.csv', index=False)

    print("\n=== Shared metrics ===")
    print(shared.to_string(index=False))
    print("\n=== Abstraction demo: reachable states ===")
    print(reachable.to_string(index=False))
    print("\n=== Verification demo: model checking ===")
    print(verification.to_string(index=False))
    print(f"\nSaved tables and figures to {out_dir}")

    assert all(contained), "Concrete trajectory left the abstract reachable set"
