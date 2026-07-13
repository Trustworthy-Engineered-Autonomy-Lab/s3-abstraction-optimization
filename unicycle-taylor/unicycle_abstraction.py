# =====================================================================
# Description: contains the necessary tools for modeling the unicycle
# system as a finite transition system with robust Taylor reachability.
# Utilizes PyModelChecking for abstraction as a Kripke structure.
# =====================================================================

# =====================================================================
# Libraries for the unicycle system
# =====================================================================

import unicycle_system_sympy as uss
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
        theta_edges,
        *,
        verbose = False,
        goal_center = np.array([40.0, 20.0]),
        goal_radius = 8.0,
        obs_center = np.array([25.0, 25.0]),
        obs_radius = 5.0
        ):
    """
    Casts the unicycle closed-loop system as a finite-state transition
    system (Kripke structure). Successor cells (abstract states) are
    assured to cover the ground-truth reachable set via 1-step Taylor
    reachability, whereby post-image AABBs are inflated with the cell's 
    Lagrange error.
    """

    # Extract model size details
    nstates_1 = len(x_edges) - 1
    nstates_2 = len(y_edges) - 1
    nstates_3 = len(theta_edges) - 1
    n_kripke_states = nstates_1 * nstates_2 * nstates_3 + 1 # includes out of bounds state
    oob_state_id = n_kripke_states - 1

    # Initialize Kripike structure components
    kripke_states = list(range(n_kripke_states))  # last state is out-of-bounds
    kripke_transitions = set()
    kripke_labels = {}
    
    # Iterate over cells and determine successors w/Taylor reachability
    if verbose:
        print("Starting abstraction construction...")
    num_states_iterated = 0
    total_states = n_kripke_states - 1
    for i in range(nstates_1):
        x_lo, x_hi = x_edges[i], x_edges[i+1]
        for j in range(nstates_2):
            y_lo, y_hi = y_edges[j], y_edges[j+1]
            for k in range(nstates_3):
                theta_lo, theta_hi = theta_edges[k], theta_edges[k+1]

                # Specify cell domain and corners
                lower_bounds = np.array([x_lo, y_lo, theta_lo])
                upper_bounds = np.array([x_hi, y_hi, theta_hi])
                all_verts = [list(combo) for combo in product(*zip(lower_bounds, upper_bounds))]
                all_verts = np.array(all_verts)

                # Determine label of current cell
                if is_goal_state(all_verts, center=goal_center, radius=goal_radius):
                    label = ['goal']
                elif is_obs_state(all_verts, center=obs_center, radius=obs_radius):
                    label = ['fail']
                else:
                    label = ['safe']

                # Compute cell centroid and evaluate Jacobian there; compute post-image AABB
                centroid = (lower_bounds + upper_bounds) / 2.0
                J = uss.jacobian(centroid)
                f_center = uss.cl_system_numeric(centroid)
                linearized_next_verts = np.array([
                    uss.linear_cl_system(vert, centroid, J=J, f_center=f_center)
                    for vert in all_verts
                ])
                next_lower_bounds = linearized_next_verts.min(axis=0)
                next_upper_bounds = linearized_next_verts.max(axis=0)

                # Compute Taylor remainder
                R_lo, R_hi = uss.taylor_remainder(lb=lower_bounds,
                                                  ub=upper_bounds)
                if abs(R_lo[2]) > 1 or abs(R_hi[2]) > 1:
                    R_lo[2] = 0.0
                    R_hi[2] = 0.0

                # Inflate AABB by Taylor remainder
                next_lower_bounds += R_lo
                next_upper_bounds += R_hi
                
                # Compute theta image interval(s) using minimal circular arc.
                if np.isfinite(next_lower_bounds[2]) and np.isfinite(next_upper_bounds[2]):
                    all_next_verts = [list(combo) for combo in product(*zip(next_lower_bounds, next_upper_bounds))]
                    all_next_verts = np.array(all_next_verts)
                    theta_intervals = theta_min_arc_intervals(all_next_verts[:, 2])
                else:
                    theta_intervals = [(theta_edges[0], theta_edges[-1])]

                # Identify successor cells
                x_min = next_lower_bounds[0]
                x_max = next_upper_bounds[0]
                y_min = next_lower_bounds[1]
                y_max = next_upper_bounds[1]
                succ_cells_set = set()
                for (theta_min, theta_max) in theta_intervals:
                    succ_cells_set.update(
                        intersecting_cells_from_aabb(
                            x_edges,
                            y_edges,
                            theta_edges,
                            x_min,
                            y_min,
                            theta_min,
                            x_max,
                            y_max,
                            theta_max,
                        )
                    )
                succ_cells = list(succ_cells_set)
                
                # Indicate if any successor goes out of bounds
                hits_oob = is_oob_state(np.stack([next_lower_bounds, next_upper_bounds]),
                                        x_bounds=(x_edges[0], x_edges[-1]),
                                        y_bounds=(y_edges[0], y_edges[-1]))

                # Allocate relations to Kripke structure components
                src = cell_to_id(i, j, k, nstates_1, nstates_2, nstates_3)
                for (ip, jp, kp) in succ_cells:
                    dst = cell_to_id(ip, jp, kp, nstates_1, nstates_2, nstates_3)
                    edge = (src, dst)
                    if edge not in kripke_transitions:
                        kripke_transitions.add(edge)
                if hits_oob:
                    edge = (src, oob_state_id)
                    if edge not in kripke_transitions:
                        kripke_transitions.add(edge)  # transitions oob

                kripke_labels[src] = label
                if verbose and num_states_iterated % 10000 == 0:
                        print(f"    > {num_states_iterated} / {total_states} states ")
                num_states_iterated += 1

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
        print(f"Final model details")
        print(f"    > States = {len(kripke_states)}")
        print(f"    > Transitions = {len(kripke_transitions)}")
        print(f"    > Avg. succ/state = {len(kripke_transitions)/len(kripke_states)}")
    
    return kripke_components

def intersecting_cells_from_aabb(
        x_params,
        y_params,
        theta_params,
        x_min,
        y_min,
        theta_min,
        x_max,
        y_max,
        theta_max,
        ) -> list[tuple[int, int, int]]:
    """
    Return all grid cell indices (i, j, k) intersected by an axis-aligned box.
    The grid is defined by bin edges:
      - x_params: length n_x + 1
      - y_params: length n_y + 1
      - theta_params: length n_theta + 1
    The AABB is [x_min, x_max] x [y_min, y_max] x [theta_min, theta_max].
    """

    x_params = np.asarray(x_params, dtype=float)
    y_params = np.asarray(y_params, dtype=float)
    theta_params = np.asarray(theta_params, dtype=float)

    nx = len(x_params) - 1
    ny = len(y_params) - 1
    ntheta = len(theta_params) - 1

    # If completely outside the overall grid bounds, no in-grid intersections.
    grid_x_min, grid_x_max = float(x_params[0]), float(x_params[-1])
    grid_y_min, grid_y_max = float(y_params[0]), float(y_params[-1])
    grid_t_min, grid_t_max = float(theta_params[0]), float(theta_params[-1])
    if (
        (x_max < grid_x_min)
        or (x_min > grid_x_max)
        or (y_max < grid_y_min)
        or (y_min > grid_y_max)
        or (theta_max < grid_t_min)
        or (theta_min > grid_t_max)
    ):
        return []

    # Use side="right" for both ends.
    # This avoids the empty-set edge case when min == max equals a grid line.
    i_lo = int(np.searchsorted(x_params, x_min, side="right") - 1)
    i_hi = int(np.searchsorted(x_params, x_max, side="right") - 1)
    j_lo = int(np.searchsorted(y_params, y_min, side="right") - 1)
    j_hi = int(np.searchsorted(y_params, y_max, side="right") - 1)
    k_lo = int(np.searchsorted(theta_params, theta_min, side="right") - 1)
    k_hi = int(np.searchsorted(theta_params, theta_max, side="right") - 1)

    i_lo = max(0, min(nx - 1, i_lo))
    i_hi = max(0, min(nx - 1, i_hi))
    j_lo = max(0, min(ny - 1, j_lo))
    j_hi = max(0, min(ny - 1, j_hi))
    k_lo = max(0, min(ntheta - 1, k_lo))
    k_hi = max(0, min(ntheta - 1, k_hi))

    if i_hi < i_lo or j_hi < j_lo or k_hi < k_lo:
        return []

    return [
        (i, j, k)
        for i in range(i_lo, i_hi + 1)
        for j in range(j_lo, j_hi + 1)
        for k in range(k_lo, k_hi + 1)
    ]


# =====================================================================
# Auxillary labeling functions
# =====================================================================

def is_goal_state(
        vertices,
        center,
        radius
        ) -> bool:
    """
    Determines whether a cell is a goal state
    """

    # Return True if all corners of the x-space cell are within the goal circle
    for v in vertices:
        if np.linalg.norm(v[:2] - center) > radius:
            return False
    return True


def is_obs_state(
        vertices,
        center,
        radius
        ) -> bool:
    """
    Determines whether a cell is an (unsafe) obstacle state
    """

    x_min = np.min(vertices[:, 0])
    x_max = np.max(vertices[:, 0])
    y_min = np.min(vertices[:, 1])
    y_max = np.max(vertices[:, 1])

    closest_x = np.clip(center[0], x_min, x_max)
    closest_y = np.clip(center[1], y_min, y_max)

    return np.linalg.norm(np.array([closest_x, closest_y]) - center) <= radius


def is_oob_state(
        vertices,
        x_bounds,
        y_bounds) -> bool:
    """
    Determines whether a cell is an out-of-bounds state
    """

    x_min, x_max = x_bounds
    y_min, y_max = y_bounds

    # Return True iff any vertex is out of bounds
    for v in vertices:
        if (v[0] < x_min) or (v[0] > x_max) or (v[1] < y_min) or (v[1] > y_max):
            return True
    return False


# =====================================================================
# Abstraction helper functions
# =====================================================================


def wrap_to_pi(angle):
    angle = np.asarray(angle)
    return (angle + np.pi) % (2 * np.pi) - np.pi


def theta_min_arc_intervals(
        thetas,
        *,
        eps=1e-12
        ) -> list[tuple[float, float]]:
    """
    Returns non-wrapping theta interval(s) in [-pi, pi] covering samples.
    Angles live on S^1. Using raw min/max on wrapped angles can incorrectly
    produce near-2*pi spans when samples straddle the -pi/pi cut.
    Returns either:
      - [(lo, hi)] with lo <= hi, or
      - [(-pi, hi), (lo, pi)] when the minimal arc wraps across the cut.
    DOMAIN HARD-CODED!
    """

    th = np.asarray(thetas, dtype=float)
    if th.size == 0:
        return [(-np.pi, np.pi)]

    # Wrap to [-pi, pi)
    th = wrap_to_pi(th)

    if th.size == 1:
        v = float(th[0])
        return [(v, v)]

    # Work on [0, 2pi) for stable circular gap computation.
    u = np.sort(th + np.pi)
    two_pi = 2.0 * np.pi

    # Find the largest gap between consecutive points on the circle.
    gaps = np.diff(np.r_[u, u[0] + two_pi])
    k = int(np.argmax(gaps))

    # Minimal covering arc is the complement of the largest gap.
    start_u = float(u[(k + 1) % u.size])
    end_u = float(u[k])
    arc_len = (end_u - start_u) % two_pi

    # Degenerate / numerical fallbacks.
    if arc_len >= two_pi - eps:
        return [(-np.pi, np.pi)]

    end_u2 = start_u + arc_len
    if end_u2 <= two_pi + eps:
        lo = start_u - np.pi
        hi = min(end_u2, two_pi) - np.pi
        return [(float(lo), float(hi))]

    # Wraps across the cut at 2pi -> split into two non-wrapping intervals.
    lo1, hi1 = start_u - np.pi, np.pi
    lo2, hi2 = -np.pi, (end_u2 - two_pi) - np.pi
    return [(float(lo2), float(hi2)), (float(lo1), float(hi1))]


# =====================================================================
# State identification helpers
# =====================================================================

def cell_to_id(i, j, k, nstates_1, nstates_2, nstates_3):
    return i * (nstates_2 * nstates_3) + j * nstates_3 + k

def id_to_cell(id, nstates_1, nstates_2, nstates_3):
    cells_per_i = nstates_2 * nstates_3
    i = id // cells_per_i
    remainder = id % cells_per_i
    j = remainder // nstates_3
    k = remainder % nstates_3
    return (i, j, k)

def id_to_bounds(id, x_edges, y_edges, theta_edges):
    i, j, k = id_to_cell(id,
                         len(x_edges)-1,
                         len(y_edges)-1,
                         len(theta_edges)-1)
    lb = np.array([x_edges[i], y_edges[j], theta_edges[k]])
    ub = np.array([x_edges[i+1], y_edges[j+1], theta_edges[k+1]])
    return lb, ub

# =====================================================================
# Initial state space initilization
# =====================================================================

def init_cells_to_ids(
        init_domain_lb,
        init_domain_ub,
        x_edges,
        y_edges,
        theta_edges,
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
        theta_edges,
        init_domain_lb[0],
        init_domain_lb[1],
        init_domain_lb[2],
        init_domain_ub[0],
        init_domain_ub[1],
        init_domain_ub[2],
    )

    nstates_1 = len(x_edges) - 1
    nstates_2 = len(y_edges) - 1
    nstates_3 = len(theta_edges) - 1

    init_ids = [
        cell_to_id(i, j, k, nstates_1, nstates_2, nstates_3)
        for (i, j, k) in init_cells]

    return init_cells, init_ids


# =====================================================================
# Functions for re-formatting the abstraction
# =====================================================================

def kripke_to_dicts(kripke_components, x_edges, y_edges, theta_edges):

    kripke_states = kripke_components['kripke_states']
    kripke_transitions = kripke_components['kripke_transitions']
    oob_state_id = (len(x_edges) - 1) * (len(y_edges) - 1) * (len(theta_edges) - 1)

    successors = {state_id: set() for state_id in kripke_states}
    cells = {}

    for state_id in kripke_states:
        if state_id == oob_state_id:
            cells[state_id] = None
        else:
            lb, ub = id_to_bounds(state_id, x_edges, y_edges, theta_edges)
            cells[state_id] = (lb, ub)

    for src, dst in kripke_transitions:
        successors[src].add(dst)

    return successors, cells


# =====================================================================
# Section for testing the above methods
# =====================================================================

if __name__ == "__main__":

    abstraction_shape = [20, 20, 20]
    domain_lb = np.array([0.0, 0.0, -np.pi])
    domain_ub = np.array([50.0, 40.0, np.pi])

    # _, kripke_components = build_abstraction(abstraction_shape, domain_lb, domain_ub, verbose=True)

    # # Save the Kripke structure components for later use
    # with open("kripke_components.pkl", "wb") as f:
    #     pkl.dump(kripke_components, f)
