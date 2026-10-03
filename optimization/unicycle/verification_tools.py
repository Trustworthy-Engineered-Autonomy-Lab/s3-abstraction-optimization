# =====================================================================
# Description: tools for verifying a pyModelChecking Kripke structure
# and analyzing the quality of verification
# =====================================================================

# =====================================================================
# Libraries for the unicycle system
# =====================================================================

import unicycle_abstraction as ua
import unicycle_objectives as uo
import unicycle_system as us
import numpy as np
import pyModelChecking as pmc
import pyModelChecking.CTL as CTL
import jax
import jax.numpy as jnp
import pickle as pkl
import time


# =====================================================================
# Kripke structure verification
# =====================================================================

def model_check_kripke(kripke_structure, log_time=False):
    """
    Returns the subset of initial states from which abstract
    trajectories satisfy the verification property. Also
    returns several verification metrics, including the sat-
    isfaction proportion (proportion of the initial states)
    that are verified.
    """

    phi = 'A (safe U goal)'
    # phi = 'A F goal'
    if log_time:
        start_time = time.perf_counter()
    sat = CTL.modelcheck(kripke_structure, phi)
    if log_time:
        end_time = time.perf_counter()
        print(f"Verif time = {end_time-start_time:.4f}s")
    sat_set = set(sat)
    init_states = list(kripke_structure.S0)

    sat_init_states = [state for state in init_states if state in sat_set]

    return sat_init_states


# =====================================================================
# Model building helpers
# =====================================================================

def build_and_verify_from_params(
    params,
    shape,
    domain_lb,
    domain_ub,
    init_domain_lb,
    init_domain_ub,
    gt_reach_fname,
    verbose=False,
    log_time=False):
    """
    Simple helper to compute verification metrics on a set of parameters.
    """

    # Extract concrete cells edges from quantization parameters
    x_edges, y_edges, theta_edges = uo.extract_grid_params(params,
                                                           shape,
                                                           domain_lb,
                                                           domain_ub)
    edges = [x_edges, y_edges, theta_edges]

    # Build the initial state set from desired bounds
    _, init_states = ua.init_cells_to_ids(init_domain_lb,
                                          init_domain_ub,
                                          x_edges,
                                          y_edges,
                                          theta_edges)

    # Build the kripke and model check
    if log_time:
        start_time = time.perf_counter()
    kripke_components = ua.build_abstraction(x_edges,
                                             y_edges,
                                             theta_edges,
                                             verbose=verbose)
    if log_time:
        end_time = time.perf_counter()
        print(f"Build time = {end_time-start_time:.4f}s")
    kripke_structure = pmc.Kripke(S=kripke_components['kripke_states'],
                                  S0=init_states,
                                  R=list(kripke_components['kripke_transitions']),
                                  L=kripke_components['kripke_labels'])
    sat_init_states = model_check_kripke(kripke_structure, log_time=log_time)

    # Classify cells formed by abstraction as "goal" or "fail" per ground truth
    with open(gt_reach_fname, "rb") as f:
        gt_reach_regions = pkl.load(f)
    ground_truth_check = check_ground_truth_fast(params,
                                                 shape,
                                                 domain_lb,
                                                 domain_ub,
                                                 gt_reach_regions)
    
    # Determine volume of the init cells that satisfy the property per ground truth
    gt_init_sat_volumes = np.array([
        v[1]
        for k, v in ground_truth_check.items()
        if v[0] == "goal" and k in init_states
    ])
    gt_sat_vol = np.sum(gt_init_sat_volumes)

    # Determine volume of init cells that satisfy the property per model checking
    mc_init_sat_volumes = np.array([
        v[1] for k, v in ground_truth_check.items()
        if v[0] == "goal" and k in sat_init_states
    ])
    mc_sat_vol = np.sum(mc_init_sat_volumes)
    recall = mc_sat_vol/gt_sat_vol
    
    return recall, kripke_components
    

# =====================================================================
# Concrete satisfaction analysis
# =====================================================================

def get_gt_reach_avoid_regions(domain_lb, domain_ub, grid_resolution=101, verbose=False):

    # Environment details
    obs_center = np.array([25.0, 25.0])
    obs_radius = 5.0
    goal_center = np.array([40.0, 20.0])
    goal_radius = 8.0

    # Unpack domain parameters
    x1_min, x2_min, x3_min = domain_lb
    x1_max, x2_max, x3_max = domain_ub

    # Initialize a uniform grid over the x-domain
    x1_vals = np.linspace(x1_min, x1_max, grid_resolution)
    x2_vals = np.linspace(x2_min, x2_max, grid_resolution)
    x3_vals = np.linspace(x3_min, x3_max, grid_resolution)

    # Iterate through each grid cell; classify as 'fail', 'goal', or 'unk'
    if verbose:
        print("Executing fixed grid reachability...")
        count = 0
    total_cells = (grid_resolution - 1) ** 3
    gt_reach_regions = {}
    for i in range(grid_resolution - 1):
        x1_lo, x1_hi = x1_vals[i], x1_vals[i+1]
        for j in range(grid_resolution - 1):
            x2_lo, x2_hi = x2_vals[j], x2_vals[j+1]
            for k in range(grid_resolution - 1):
                x3_lo, x3_hi = x3_vals[k], x3_vals[k+1]

                # Define cell corners
                verts = np.array(
                    [
                        [x1_lo, x2_lo, x3_lo],
                        [x1_lo, x2_hi, x3_lo],
                        [x1_hi, x2_hi, x3_lo],
                        [x1_hi, x2_lo, x3_lo],
                        [x1_lo, x2_lo, x3_hi],
                        [x1_lo, x2_hi, x3_hi],
                        [x1_hi, x2_hi, x3_hi],
                        [x1_hi, x2_lo, x3_hi],
                    ]
                )

                # Push vertices through dynamics until any fail or all succeed
                max_steps = 10_000
                reached_terminal = False
                for _ in range(max_steps):

                    if ua.is_obs_state(verts, center=obs_center, radius=obs_radius):
                        gt_reach_regions[(i, j, k)] = 'fail'
                        reached_terminal = True
                        break

                    if ua.is_goal_state(verts, goal_center, goal_radius):
                        gt_reach_regions[(i, j, k)] = 'goal'
                        reached_terminal = True
                        break

                    if ua.is_oob_state(verts, x_bounds=(x1_min, x1_max), y_bounds=(x2_min, x2_max)):
                        gt_reach_regions[(i, j, k)] = 'fail'
                        reached_terminal = True
                        break

                    verts = np.array([us.cl_system(vert) for vert in verts])

                if not reached_terminal:
                    gt_reach_regions[(i, j, k)] = 'unk'
                
                if verbose and (count % 1000 == 0):
                    print(f"    > Processed {count} / {total_cells} regions...")
                count += 1
    
    return gt_reach_regions


def get_gt_reach_regions(domain_lb, domain_ub, grid_resolution=21, verbose=False):

    # Environment details
    obs_center = np.array([25.0, 25.0])
    obs_radius = 5.0
    goal_center = np.array([40.0, 20.0])
    goal_radius = 8.0

    # Unpack domain parameters
    x1_min, x2_min, x3_min = domain_lb
    x1_max, x2_max, x3_max = domain_ub

    # Initialize a uniform grid over the x-domain
    x1_vals = np.linspace(x1_min, x1_max, grid_resolution)
    x2_vals = np.linspace(x2_min, x2_max, grid_resolution)
    x3_vals = np.linspace(x3_min, x3_max, grid_resolution)

    # Iterate through each grid cell; classify as 'fail', 'goal', or 'unk'
    if verbose:
        print("Executing fixed grid reachability...")
        count = 0
    total_cells = (grid_resolution - 1) ** 3
    gt_reach_regions = {}
    for i in range(grid_resolution - 1):
        x1_lo, x1_hi = x1_vals[i], x1_vals[i+1]
        for j in range(grid_resolution - 1):
            x2_lo, x2_hi = x2_vals[j], x2_vals[j+1]
            for k in range(grid_resolution - 1):
                x3_lo, x3_hi = x3_vals[k], x3_vals[k+1]

                # Define cell corners
                verts = np.array(
                    [
                        [x1_lo, x2_lo, x3_lo],
                        [x1_lo, x2_hi, x3_lo],
                        [x1_hi, x2_hi, x3_lo],
                        [x1_hi, x2_lo, x3_lo],
                        [x1_lo, x2_lo, x3_hi],
                        [x1_lo, x2_hi, x3_hi],
                        [x1_hi, x2_hi, x3_hi],
                        [x1_hi, x2_lo, x3_hi],
                    ]
                )

                # Push vertices through dynamics until any fail or all succeed
                max_steps = 5_000
                reached_terminal = False
                for _ in range(max_steps):

                    if ua.is_goal_state(verts, goal_center, goal_radius):
                        gt_reach_regions[(i, j, k)] = 'goal'
                        reached_terminal = True
                        break

                    verts = np.array([us.cl_system(vert) for vert in verts])

                if not reached_terminal:
                    gt_reach_regions[(i, j, k)] = 'unk'
                
                if verbose and (count % 1000 == 0):
                    print(f"    > Processed {count} / {total_cells} regions...")
                count += 1
    
    return gt_reach_regions


def check_ground_truth_fast(
        params,
        shape,
        domain_lb,
        domain_ub,
        gt_reach_regions,
        edges=None
        ):

    """
    Classify abstraction cells according to "concrete" satisfaction. Returns a dict that
    labels each cell (by its scalar cell ID) as either "goal" or "fail".
    If edges = (x_edges, y_edges, theta_edges) is given, it is used instead of
    deriving the grid from params.
    """

    if not gt_reach_regions:
        raise ValueError("gt_reach_regions is empty")

    # Unpack domain parameters
    x1_min, x2_min, x3_min = domain_lb
    x1_max, x2_max, x3_max = domain_ub

    if edges is None:
        x_edges, y_edges, theta_edges = uo.extract_grid_params(params, shape, domain_lb, domain_ub)
    else:
        x_edges, y_edges, theta_edges = edges

    # Infer fixed grid resolution per axis from keys.
    sample_key = next(iter(gt_reach_regions.keys()))
    if not (isinstance(sample_key, tuple) and len(sample_key) == 3):
        raise ValueError("Expected gt_reach_regions keys of form (i, j, k)")

    max_i = max(int(k[0]) for k in gt_reach_regions.keys())
    max_j = max(int(k[1]) for k in gt_reach_regions.keys())
    max_k = max(int(k[2]) for k in gt_reach_regions.keys())
    res_1 = max_i + 2
    res_2 = max_j + 2
    res_3 = max_k + 2

    x1_vals = np.linspace(x1_min, x1_max, res_1)
    x2_vals = np.linspace(x2_min, x2_max, res_2)
    x3_vals = np.linspace(x3_min, x3_max, res_3)

    nstates_1 = len(x_edges) - 1
    nstates_2 = len(y_edges) - 1
    nstates_3 = len(theta_edges) - 1

    goal_cells = {idx for (idx, lab) in gt_reach_regions.items() if lab == "goal"}

    def _fixed_index_range(edges, lo, hi):
        # All fixed cells whose interval intersects [lo, hi].
        # Use side="left" for upper bound so hi on an edge doesn't include the cell to the right.
        n = len(edges) - 1
        if hi <= edges[0] or lo >= edges[-1]:
            return None
        i_lo = int(np.searchsorted(edges, lo, side="right") - 1)
        i_hi = int(np.searchsorted(edges, hi, side="left") - 1)
        i_lo = max(0, min(n - 1, i_lo))
        i_hi = max(0, min(n - 1, i_hi))
        if i_hi < i_lo:
            return None
        return i_lo, i_hi

    ground_truth_check = {}

    for i in range(nstates_1):
        a1_lo, a1_hi = float(x_edges[i]), float(x_edges[i + 1])
        for j in range(nstates_2):
            a2_lo, a2_hi = float(y_edges[j]), float(y_edges[j + 1])
            for k in range(nstates_3):
                a3_lo, a3_hi = float(theta_edges[k]), float(theta_edges[k + 1])

                src = ua.cell_to_id(i, j, k, nstates_1, nstates_2, nstates_3)
                cell_vol = (a1_hi - a1_lo) * (a2_hi - a2_lo) * (a3_hi - a3_lo)

                # If abstraction cell is out of the domain, mark fail.
                if (
                    (a1_lo < x1_min)
                    or (a1_hi > x1_max)
                    or (a2_lo < x2_min)
                    or (a2_hi > x2_max)
                    or (a3_lo < x3_min)
                    or (a3_hi > x3_max)
                ):
                    ground_truth_check[src] = ["fail", cell_vol]
                    continue

                r1 = _fixed_index_range(x1_vals, a1_lo, a1_hi)
                r2 = _fixed_index_range(x2_vals, a2_lo, a2_hi)
                r3 = _fixed_index_range(x3_vals, a3_lo, a3_hi)
                if r1 is None or r2 is None or r3 is None:
                    ground_truth_check[src] = ["fail", cell_vol]
                    continue

                i_lo, i_hi = r1
                j_lo, j_hi = r2
                k_lo, k_hi = r3

                # 'goal' iff ALL overlapping fixed cells are goal.
                all_goal = True
                for ii in range(i_lo, i_hi + 1):
                    for jj in range(j_lo, j_hi + 1):
                        for kk in range(k_lo, k_hi + 1):
                            if (ii, jj, kk) not in goal_cells:
                                all_goal = False
                                break
                        if not all_goal:
                            break
                    if not all_goal:
                        break

                ground_truth_check[src] = ["goal", cell_vol] if all_goal else ["fail", cell_vol]

    return ground_truth_check


# =====================================================================
# Section to test above methods
# =====================================================================

def plot_verification_slice(
        sat_init_states,
        init_states,
        ground_truth_check,
        x_edges,
        y_edges,
        theta_edges,
        save_path,
        theta_slice=0.0,
        show=False):
    """
    Colors the initial cells of one theta slice by verification outcome.
    """

    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    n1, n2, n3 = len(x_edges) - 1, len(y_edges) - 1, len(theta_edges) - 1
    k = int(np.clip(np.searchsorted(theta_edges, theta_slice, side='right') - 1, 0, n3 - 1))
    init_set, sat_set = set(init_states), set(sat_init_states)

    # 0: not initial, 1: verified, 2: gt-satisfying but missed, 3: gt-failing
    grid = np.zeros((n2, n1))
    for i in range(n1):
        for j in range(n2):
            cid = ua.cell_to_id(i, j, k, n1, n2, n3)
            if cid not in init_set:
                continue
            if cid in sat_set:
                grid[j, i] = 1
            elif ground_truth_check[cid][0] == "goal":
                grid[j, i] = 2
            else:
                grid[j, i] = 3

    colors = ['#eeeeee', '#2ca02c', '#ff7f0e', '#d62728']
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.pcolormesh(x_edges, y_edges, grid, cmap=ListedColormap(colors), vmin=0, vmax=3)
    ax.add_patch(plt.Circle((40.0, 20.0), 8.0, fill=False, ec='green', lw=2))
    ax.add_patch(plt.Circle((25.0, 25.0), 5.0, fill=False, ec='black', lw=2))
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Verification of A(safe U goal), '
                 f'theta in [{theta_edges[k]:.2f}, {theta_edges[k+1]:.2f}]')
    ax.legend(handles=[
        Patch(color=colors[0], label='not an initial cell'),
        Patch(color=colors[1], label='verified'),
        Patch(color=colors[2], label='ground-truth goal, not verified (conservatism)'),
        Patch(color=colors[3], label='ground-truth fail'),
    ], loc='upper center', bbox_to_anchor=(0.5, -0.12), fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":

    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Build the unicycle abstraction on a uniform grid, "
                    "model check A(safe U goal), and report the results."
    )
    parser.add_argument("--shape", type=int, default=20)
    parser.add_argument("--show", action="store_true")
    cli_args = parser.parse_args()

    # Fixed abstraction and environment settings
    abstraction_shape = [cli_args.shape] * 3
    domain_lb = np.array([0.0, 0.0, -np.pi])
    domain_ub = np.array([50.0, 50.0, np.pi])
    init_domain_lb = np.array([0.0, 0.0, -np.pi/4])
    init_domain_ub = np.array([50.0, 50.0, np.pi/4])
    params = jnp.zeros(sum(abstraction_shape))  # zero gap-params: uniform grid

    base_dir = Path(__file__).parent / 'artifacts'
    gt_reach_fname = base_dir / 'cache' / 'reach.pkl'

    # Ground-truth reachability, cached on disk
    if gt_reach_fname.exists():
        with open(gt_reach_fname, "rb") as f:
            gt_reach_regions = pkl.load(f)
    else:
        gt_reach_regions = get_gt_reach_regions(domain_lb, domain_ub, verbose=True)
        gt_reach_fname.parent.mkdir(parents=True, exist_ok=True)
        with open(gt_reach_fname, "wb") as f:
            pkl.dump(gt_reach_regions, f)

    # Build the abstraction
    x_edges, y_edges, theta_edges = [
            np.linspace(lb, ub, n + 1)
            for lb, ub, n in zip(domain_lb, domain_ub, abstraction_shape)
        ]
    _, init_states = ua.init_cells_to_ids(
        init_domain_lb, init_domain_ub, x_edges, y_edges, theta_edges)
    t0 = time.perf_counter()
    kripke_components = ua.build_abstraction(
        x_edges, y_edges, theta_edges, verbose=True)
    build_time = time.perf_counter() - t0
    kripke_structure = pmc.Kripke(
        S=kripke_components['kripke_states'],
        S0=init_states,
        R=list(kripke_components['kripke_transitions']),
        L=kripke_components['kripke_labels'])

    # Model check
    t0 = time.perf_counter()
    sat_init_states = model_check_kripke(kripke_structure)
    check_time = time.perf_counter() - t0

    # Compare against ground truth by initial-state volume
    ground_truth_check = check_ground_truth_fast(
        params, abstraction_shape, domain_lb, domain_ub, gt_reach_regions,
        edges=(x_edges, y_edges, theta_edges))
    init_set, sat_set = set(init_states), set(sat_init_states)
    init_vol = sum(ground_truth_check[s][1] for s in init_states)
    mc_vol = sum(ground_truth_check[s][1] for s in sat_init_states)
    gt_sat = [s for s in init_states if ground_truth_check[s][0] == "goal"]
    gt_vol = sum(ground_truth_check[s][1] for s in gt_sat)
    unsound = [s for s in sat_init_states if ground_truth_check[s][0] != "goal"]

    stats = ua.summarize_abstraction(kripke_components)
    print("Model statistics")
    print(f"  States:                    {stats['states']:,}")
    print(f"  Transitions:               {stats['transitions']:,}")
    print(f"  Out-degree (mean/min/max): {stats['out_degree_mean']:.2f} / "
          f"{stats['out_degree_min']} / {stats['out_degree_max']}")
    print(f"  Cell self-loops:           {stats['cell_self_loops']:,}")
    print(f"  Edges into OOB state:      {stats['edges_to_oob']:,}")
    print(f"  Labels:                    {stats['label_counts']}")
    print(f"  Build time:                {build_time:.2f}s")

    print("\nModel checking: A (safe U goal)")
    print(f"  Initial cells:             {len(init_set):,}")
    print(f"  Verified initial cells:    {len(sat_set):,} "
          f"({len(sat_set) / len(init_set):.1%})")
    print(f"  Ground-truth goal cells:   {len(gt_sat):,} "
          f"({len(gt_sat) / len(init_set):.1%})")
    print(f"  Verified volume fraction:  {mc_vol / init_vol:.1%}")
    print(f"  Recall (vs. ground truth): "
          f"{mc_vol / gt_vol if gt_vol > 0 else float('nan'):.4f}")
    print(f"  Verified, GT not 'goal':   {len(unsound)} "
          f"(goal-labeled cells at the goal boundary that overlap a coarser GT cell)")
    print(f"  Model check time:          {check_time:.2f}s")

    out_dir = base_dir / 'demo'
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / 'verification_slice.png'
    plot_verification_slice(
        sat_init_states, init_states, ground_truth_check,
        x_edges, y_edges, theta_edges, save_path, show=cli_args.show)
    print(f"\nSaved figure to {save_path}")
