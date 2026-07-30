# =====================================================================
# Description: contains the necessary tools for modeling the MountainCar
# system as a finite transition system with robust Taylor reachability.
# Utilizes PyModelChecking for abstraction as a Kripke structure.
# =====================================================================

# =====================================================================
# Libraries for the unicycle system
# =====================================================================

import mountain_car_system as mcs
import pyModelChecking as pmc
import numpy as np
from itertools import product
import pickle as pkl


# =====================================================================
# Primary functions for casting system as a Kripke structure
# =====================================================================


def build_abstraction(
        x_edges,
        y_edges,
        *,
        verbose = False,
        strict_derivatives = False,
        derivative_boundary_inset = 1e-9,
        approximate_hessian_resolution = 3,
    ):
    """
    Cast the closed-loop Mountain Car system as a finite-state transition
    system. 
    """

    # Extract model size details
    nstates_1 = len(x_edges) - 1
    nstates_2 = len(y_edges) - 1
    n_kripke_states = nstates_1 * nstates_2 + 1
    oob_state_id = n_kripke_states - 1

    # Initialize Kripike structure components
    kripke_states = list(range(n_kripke_states))  # last state is out-of-bounds
    kripke_transitions = set()
    kripke_labels = {}
    
    # Iterate over cells and determine successors w/Taylor reachability
    num_states_iterated = 0
    total_states = n_kripke_states - 1
    progress_interval = max(1, total_states // 10)
    if verbose:
        print("\nBuilding Mountain Car abstraction")
        print(f"  Grid: {nstates_1} x {nstates_2} ({total_states:,} cells)")
        print("  Progress: 0.0%")
    for i in range(nstates_1):
        x_lo, x_hi = x_edges[i], x_edges[i+1]
        for j in range(nstates_2):
            y_lo, y_hi = y_edges[j], y_edges[j+1]

            # Specify cell domain and corners
            lower_bounds = np.array([x_lo, y_lo])
            upper_bounds = np.array([x_hi, y_hi])
            all_verts = [list(combo) for combo in product(*zip(lower_bounds, upper_bounds))]
            all_verts = np.array(all_verts)

            # Determine label of current cell
            if is_goal_state(all_verts,):
                label = ['goal']
            else:
                label = ['safe']

            interval_lower, interval_upper = mcs.interval_cl_system(
                lower_bounds,
                upper_bounds,
            )
            try:
                centroid = (lower_bounds + upper_bounds) / 2.0
                J = mcs.jacobian(centroid)
                f_center = mcs.cl_system_numeric(centroid)
                linearized_next_verts = np.array([
                    mcs.linear_cl_system(
                        vert,
                        centroid,
                        J=J,
                        f_center=f_center,
                    )
                    for vert in all_verts
                ])
                next_lower_bounds = linearized_next_verts.min(axis=0)
                next_upper_bounds = linearized_next_verts.max(axis=0)
                R_lo, R_hi = mcs.taylor_remainder(
                    lower_bounds,
                    upper_bounds,
                    strict=True,
                )
            except mcs.DerivativeDomainError:
                if strict_derivatives:
                    raise
                next_lower_bounds = interval_lower
                next_upper_bounds = interval_upper
            else:
                taylor_lower = next_lower_bounds + R_lo
                taylor_upper = next_upper_bounds + R_hi
                next_lower_bounds = np.maximum(taylor_lower, interval_lower)
                next_upper_bounds = np.minimum(taylor_upper, interval_upper)
                if np.any(next_lower_bounds > next_upper_bounds):
                    next_lower_bounds = interval_lower
                    next_upper_bounds = interval_upper

            # Identify successor cells
            x_min = next_lower_bounds[0]
            x_max = next_upper_bounds[0]
            y_min = next_lower_bounds[1]
            y_max = next_upper_bounds[1]
            succ_cells = intersecting_cells_from_aabb(
                        x_edges,
                        y_edges,
                        x_min,
                        y_min,
                        x_max,
                        y_max)

            # Allocate relations to Kripke structure components
            src = cell_to_id(i, j, nstates_1, nstates_2)
            for (ip, jp) in succ_cells:
                dst = cell_to_id(ip, jp, nstates_1, nstates_2)
                edge = (src, dst)
                if edge not in kripke_transitions:
                    kripke_transitions.add(edge)

            kripke_labels[src] = label
            num_states_iterated += 1
            if verbose and (
                num_states_iterated % progress_interval == 0
                or num_states_iterated == total_states
            ):
                progress = 100 * num_states_iterated / total_states
                print(
                    f"  Progress: {num_states_iterated:,} / {total_states:,} "
                    f"cells ({progress:5.1f}%)"
                )

    # Label the out-of-bounds state; add self-loop
    kripke_labels[oob_state_id] = ['fail']
    kripke_transitions.add((oob_state_id, oob_state_id))

    # Output as dictionary
    kripke_components = {
        'kripke_states': kripke_states,
        'kripke_transitions': kripke_transitions,
        'kripke_labels': kripke_labels}
    
    # Final model summary
    if verbose:
        print("Abstraction complete")
        print(f"  States:             {len(kripke_states):,}")
        print(f"  Transitions:        {len(kripke_transitions):,}")
        print(
            f"  Average successors: "
            f"{len(kripke_transitions) / len(kripke_states):.2f}\n"
        )
    
    return kripke_components

def intersecting_cells_from_aabb(
        x_params,
        y_params,
        x_min,
        y_min,
        x_max,
        y_max,
    ) -> list[tuple[int, int]]:
    """
    Return all grid cell indices (i, j) intersected by an axis-aligned box.
    The grid is defined by bin edges:
      - x_params: length n_x + 1
      - y_params: length n_y + 1
    The AABB is [x_min, x_max] x [y_min, y_max].
    """

    x_params = np.asarray(x_params, dtype=float)
    y_params = np.asarray(y_params, dtype=float)

    nx = len(x_params) - 1
    ny = len(y_params) - 1

    # If completely outside the overall grid bounds, no in-grid intersections.
    grid_x_min, grid_x_max = float(x_params[0]), float(x_params[-1])
    grid_y_min, grid_y_max = float(y_params[0]), float(y_params[-1])
    if (
        (x_max < grid_x_min)
        or (x_min > grid_x_max)
        or (y_max < grid_y_min)
        or (y_min > grid_y_max)
    ):
        return []

    # Use side="right" for both ends.
    # This avoids the empty-set edge case when min == max equals a grid line.
    i_lo = int(np.searchsorted(x_params, x_min, side="right") - 1)
    i_hi = int(np.searchsorted(x_params, x_max, side="right") - 1)
    j_lo = int(np.searchsorted(y_params, y_min, side="right") - 1)
    j_hi = int(np.searchsorted(y_params, y_max, side="right") - 1)

    i_lo = max(0, min(nx - 1, i_lo))
    i_hi = max(0, min(nx - 1, i_hi))
    j_lo = max(0, min(ny - 1, j_lo))
    j_hi = max(0, min(ny - 1, j_hi))

    if i_hi < i_lo or j_hi < j_lo:
        return []

    return [
        (i, j)
        for i in range(i_lo, i_hi + 1)
        for j in range(j_lo, j_hi + 1)
    ]


# =====================================================================
# Auxillary labeling functions
# =====================================================================

def is_goal_state(vertices):
    """
    Are all vertices in the goal set?
    """
    for v in vertices:
        if v[0] < 0.45:
            return False
    return True

# =====================================================================
# State identification helpers
# =====================================================================

def cell_to_id(i, j, nstates_1, nstates_2):
    return i * nstates_2 + j

def id_to_cell(id, nstates_1, nstates_2):
    i = id // nstates_2
    j = id % nstates_2
    return (i, j)

def id_to_bounds(id, x_edges, y_edges):
    i, j = id_to_cell(id, len(x_edges)-1, len(y_edges)-1)
    lb = np.array([x_edges[i], y_edges[j]])
    ub = np.array([x_edges[i+1], y_edges[j+1]])
    return lb, ub

# =====================================================================
# Initial state space initilization
# =====================================================================

def init_cells_to_ids(
        init_domain_lb,
        init_domain_ub,
        x_edges,
        y_edges,
        ) -> list[int]:
    """
    Return all abstract state ids whose cells overlap the given initial
    axis-aligned subdomain of the continuous state space.
    """

    init_domain_lb = np.asarray(init_domain_lb, dtype=float)
    init_domain_ub = np.asarray(init_domain_ub, dtype=float)

    init_cells = intersecting_cells_from_aabb(
        x_edges,
        y_edges,
        init_domain_lb[0],
        init_domain_lb[1],
        init_domain_ub[0],
        init_domain_ub[1],
    )

    nstates_1 = len(x_edges) - 1
    nstates_2 = len(y_edges) - 1

    init_ids = [
        cell_to_id(i, j, nstates_1, nstates_2)
        for (i, j) in init_cells]

    return init_cells, init_ids


# =====================================================================
# Functions for re-formatting the abstraction
# =====================================================================

def kripke_to_dicts(kripke_components, x_edges, y_edges):

    kripke_states = kripke_components['kripke_states']
    kripke_transitions = kripke_components['kripke_transitions']
    oob_state_id = (len(x_edges) - 1) * (len(y_edges) - 1)

    successors = {state_id: set() for state_id in kripke_states}
    cells = {}

    for state_id in kripke_states:
        if state_id == oob_state_id:
            cells[state_id] = None
        else:
            lb, ub = id_to_bounds(state_id, x_edges, y_edges)
            cells[state_id] = (lb, ub)

    for src, dst in kripke_transitions:
        successors[src].add(dst)

    return successors, cells


# =====================================================================
# Section for testing the above methods
# =====================================================================

if __name__ == "__main__":

    abstraction_shape = [4, 4]
    domain_lb = np.array([-10.0, -10.0])
    domain_ub = np.array([10.0, 10.0])

    x_edges = np.linspace(domain_lb[0], domain_ub[0], abstraction_shape[0]+1)
    y_edges = np.linspace(domain_lb[1], domain_ub[1], abstraction_shape[1]+1)

    kripke_components = build_abstraction(x_edges, y_edges, verbose=True)

    successors, cells = kripke_to_dicts(kripke_components, x_edges, y_edges)
    print(cells)
    print(successors)
