# =====================================================================
# Description: tools for verifying a pyModelChecking Kripke structure
# and analyzing the quality of verification
# =====================================================================

# =====================================================================
# Libraries for the unicycle system
# =====================================================================

import mountain_car_abstraction as mca
import mountain_car_system as mcs
import mountain_car_objectives as mco
import numpy as np
import pyModelChecking as pmc
import pyModelChecking.CTL as CTL
import jax
import jax.numpy as jnp
import pickle as pkl
import time
from itertools import product
from pathlib import Path

XSTAR = np.array([5.0, 5.0])


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

    phi = 'A F goal'
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
    x_edges, y_edges = mco.extract_grid_params(params,
                                                shape,
                                                domain_lb,
                                                domain_ub)

    # Build the initial state set from desired bounds
    _, init_states = mca.init_cells_to_ids(init_domain_lb,
                                          init_domain_ub,
                                          x_edges,
                                          y_edges)

    # Build the kripke and model check
    if log_time:
        start_time = time.perf_counter()
    kripke_components = mca.build_abstraction(x_edges,
                                             y_edges,
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

def get_gt_reach_regions(domain_lb, domain_ub, grid_resolution=101, verbose=False):

    # Unpack domain parameters
    x1_min, x2_min = domain_lb
    x1_max, x2_max = domain_ub

    # Initialize a uniform grid over the x-domain
    x1_vals = np.linspace(x1_min, x1_max, grid_resolution)
    x2_vals = np.linspace(x2_min, x2_max, grid_resolution)

    # Iterate through each grid cell; classify as 'fail', 'goal', or 'unk'
    if verbose:
        print("Executing fixed grid reachability...")
        count = 0
    total_cells = (grid_resolution - 1) ** 2

    gt_reach_regions = {}
    for i in range(grid_resolution - 1):
        x1_lo, x1_hi = x1_vals[i], x1_vals[i+1]
        for j in range(grid_resolution - 1):
            x2_lo, x2_hi = x2_vals[j], x2_vals[j+1]

            # Define cell corners
            lower_bounds = np.array([x1_lo, x2_lo])
            upper_bounds = np.array([x1_hi, x2_hi])
            verts = np.array(
                [list(combo) for combo in product(*zip(lower_bounds, upper_bounds))]
            )

            # Push vertices through dynamics until any fail or all succeed
            max_steps = 10_000
            reached_terminal = False
            for _ in range(max_steps):

                if mca.is_goal_state(verts):
                    gt_reach_regions[(i, j)] = 'goal'
                    reached_terminal = True
                    break

                verts = np.array([mcs.cl_system_numeric(vert) for vert in verts])

            if not reached_terminal:
                gt_reach_regions[(i, j)] = 'unk'
            
            if verbose and (count % 1000 == 0):
                print(f"    > Processed {count} / {total_cells} regions...")
            count += 1

    return gt_reach_regions


def check_ground_truth_fast(
        params,
        shape,
        domain_lb,
        domain_ub,
        gt_reach_regions
        ):

    """
    Classify abstraction cells according to "concrete" satisfaction. Returns a dict that
    labels each cell (by its scalar cell ID) as either "goal" or "fail".
    """

    if not gt_reach_regions:
        raise ValueError("gt_reach_regions is empty")

    # Unpack domain parameters
    x1_min, x2_min = domain_lb
    x1_max, x2_max = domain_ub

    x_edges, y_edges = mco.extract_grid_params(params, shape, domain_lb, domain_ub)

    # Infer fixed grid resolution per axis from keys.
    sample_key = next(iter(gt_reach_regions.keys()))
    if not (isinstance(sample_key, tuple) and len(sample_key) == 2):
        raise ValueError("Expected gt_reach_regions keys of form (i, j)")

    max_i = max(int(k[0]) for k in gt_reach_regions.keys())
    max_j = max(int(k[1]) for k in gt_reach_regions.keys())
    res_1 = max_i + 2
    res_2 = max_j + 2

    x1_vals = np.linspace(x1_min, x1_max, res_1)
    x2_vals = np.linspace(x2_min, x2_max, res_2)

    nstates_1 = len(x_edges) - 1
    nstates_2 = len(y_edges) - 1

    goal_cells = {idx for (idx, lab) in gt_reach_regions.items() if lab == "goal"}

    def _fixed_index_range(edges, lo, hi):
        # All fixed cells whose interval intersects [lo, hi].
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
            src = mca.cell_to_id(i, j, nstates_1, nstates_2)
            cell_area = (a1_hi - a1_lo) * (a2_hi - a2_lo)

            # If abstraction cell is out of the domain, mark fail.
            if (
                (a1_lo < x1_min)
                or (a1_hi > x1_max)
                or (a2_lo < x2_min)
                or (a2_hi > x2_max)
            ):
                ground_truth_check[src] = ["fail", cell_area]
                continue

            r1 = _fixed_index_range(x1_vals, a1_lo, a1_hi)
            r2 = _fixed_index_range(x2_vals, a2_lo, a2_hi)
            if r1 is None or r2 is None:
                ground_truth_check[src] = ["fail", cell_area]
                continue

            i_lo, i_hi = r1
            j_lo, j_hi = r2

            # 'goal' iff ALL overlapping fixed cells are goal.
            all_goal = True
            for ii in range(i_lo, i_hi + 1):
                for jj in range(j_lo, j_hi + 1):
                    if (ii, jj) not in goal_cells:
                        all_goal = False
                        break
                if not all_goal:
                    break

            ground_truth_check[src] = ["goal", cell_area] if all_goal else ["fail", cell_area]

    return ground_truth_check


# =====================================================================
# Section to test above methods
# =====================================================================

if __name__ == "__main__":

    # Fixed abstraction and environment settings
    abstraction_shape = [100, 100]
    domain_lb = np.array([-1.2, -0.07])
    domain_ub = np.array([0.6, 0.07])

    gt_reach_regions = get_gt_reach_regions(domain_lb, domain_ub, verbose=True)

    output_path = Path(__file__).with_name("mc_reach_regions.pkl")
    with output_path.open("wb") as f:
        pkl.dump(gt_reach_regions, f)
