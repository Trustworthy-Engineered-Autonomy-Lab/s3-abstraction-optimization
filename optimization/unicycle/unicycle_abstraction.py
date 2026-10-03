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
    num_states_iterated = 0
    total_states = n_kripke_states - 1
    progress_interval = max(1, total_states // 10)
    if verbose:
        print("\nBuilding unicycle abstraction")
        print(f"  Grid: {nstates_1} x {nstates_2} x {nstates_3} ({total_states:,} cells)")
        print("  Progress: 0.0%")
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
    """

    th = np.asarray(thetas, dtype=float)
    if th.size == 0:
        return [(-np.pi, np.pi)]

    # Wrap to [-pi, pi)
    th = wrap_to_pi(th)

    if th.size == 1:
        v = float(th[0])
        return [(v, v)]

    # Work on [0, 2pi) for stable circular gap computation
    u = np.sort(th + np.pi)
    two_pi = 2.0 * np.pi

    # Find the largest gap between consecutive points on the circle
    gaps = np.diff(np.r_[u, u[0] + two_pi])
    k = int(np.argmax(gaps))

    # Minimal covering arc is the complement of the largest gap
    start_u = float(u[(k + 1) % u.size])
    end_u = float(u[k])
    arc_len = (end_u - start_u) % two_pi

    # Degenerate / numerical fallbacks
    if arc_len >= two_pi - eps:
        return [(-np.pi, np.pi)]

    end_u2 = start_u + arc_len
    if end_u2 <= two_pi + eps:
        lo = start_u - np.pi
        hi = min(end_u2, two_pi) - np.pi
        return [(float(lo), float(hi))]

    # Wraps across the cut at 2pi -> split into two non-wrapping intervals
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

def summarize_abstraction(kripke_components):
    """
    Computes basic size and structure statistics of the abstraction.
    """

    states = kripke_components['kripke_states']
    transitions = kripke_components['kripke_transitions']
    labels = kripke_components['kripke_labels']
    oob_id = states[-1]

    out_degree = {s: 0 for s in states}
    self_loops = 0
    oob_edges = 0
    for src, dst in transitions:
        out_degree[src] += 1
        if src == dst and src != oob_id:
            self_loops += 1
        if dst == oob_id and src != oob_id:
            oob_edges += 1
    degrees = np.array(list(out_degree.values()))

    label_counts = {}
    for s in states:
        for lab in labels[s]:
            label_counts[lab] = label_counts.get(lab, 0) + 1

    return {
        'states': len(states),
        'transitions': len(transitions),
        'out_degree_mean': float(degrees.mean()),
        'out_degree_min': int(degrees.min()),
        'out_degree_max': int(degrees.max()),
        'deadlock_states': int((degrees == 0).sum()),
        'cell_self_loops': self_loops,
        'edges_to_oob': oob_edges,
        'label_counts': label_counts,
    }


def locate_cell(state, x_edges, y_edges, theta_edges):
    """
    Returns the id of the cell containing a concrete state, or the
    out-of-bounds id if the position leaves the (x, y) domain.
    """

    n1, n2, n3 = len(x_edges) - 1, len(y_edges) - 1, len(theta_edges) - 1
    x, y, th = state
    if not (x_edges[0] <= x <= x_edges[-1] and y_edges[0] <= y <= y_edges[-1]):
        return n1 * n2 * n3
    th = float(wrap_to_pi(th))
    i = int(np.clip(np.searchsorted(x_edges, x, side='right') - 1, 0, n1 - 1))
    j = int(np.clip(np.searchsorted(y_edges, y, side='right') - 1, 0, n2 - 1))
    k = int(np.clip(np.searchsorted(theta_edges, th, side='right') - 1, 0, n3 - 1))
    return cell_to_id(i, j, k, n1, n2, n3)


def simulate_concrete(x0, horizon):
    """
    Rolls out the concrete closed-loop system; returns (horizon+1, 3).
    """

    traj = [np.asarray(x0, dtype=float)]
    for _ in range(horizon):
        traj.append(uss.cl_system_numeric(traj[-1]))
    return np.array(traj)


def abstract_frontiers(successors, start_id, horizon):
    """
    Exact set of abstract states reachable at each step from start_id.
    """

    frontiers = [{start_id}]
    for _ in range(horizon):
        nxt = set()
        for q in frontiers[-1]:
            nxt |= successors[q]
        frontiers.append(nxt)
    return frontiers


def sample_abstract_rollouts(successors, start_id, horizon, n_rollouts, rng):
    """
    Random abstract paths (one uniformly chosen successor per step).
    """

    paths = []
    for _ in range(n_rollouts):
        path = [start_id]
        for _ in range(horizon):
            path.append(int(rng.choice(sorted(successors[path[-1]]))))
        paths.append(path)
    return paths


def plot_overlay(traj, frontiers, rollouts, cells, x_edges, y_edges, *,
                 goal_center, goal_radius, obs_center, obs_radius,
                 save_path, show=True):
    """
    Concrete trajectory over the abstract reachable cells (x-y plane).
    """

    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    horizon = len(frontiers) - 1
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 6),
                                  gridspec_kw={'width_ratios': [2, 1]})
    cmap = plt.get_cmap("viridis")

    # Project each frontier cell to its (x, y) footprint, one patch per step
    for t, frontier in enumerate(frontiers):
        footprints = {
            (cells[q][0][0], cells[q][0][1], cells[q][1][0], cells[q][1][1])
            for q in frontier if cells[q] is not None
        }
        for (x0, y0, x1, y1) in footprints:
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                   facecolor=cmap(t / max(horizon, 1)),
                                   alpha=0.12, edgecolor='none'))

    for path in rollouts:
        pts = np.array([[(cells[q][0][0] + cells[q][1][0]) / 2,
                         (cells[q][0][1] + cells[q][1][1]) / 2]
                        if cells[q] is not None else [np.nan, np.nan]
                        for q in path])
        ax.plot(pts[:, 0], pts[:, 1], '-', color='gray', lw=0.7, alpha=0.6)

    ax.add_patch(Circle(goal_center, goal_radius, fill=False, ec='green', lw=2))
    ax.add_patch(Circle(obs_center, obs_radius, fill=False, ec='red', lw=2))
    ax.plot(traj[:, 0], traj[:, 1], 'k.-', lw=2, label='concrete trajectory')
    ax.plot(*traj[0, :2], 'ko', ms=9, mfc='none', label='start')
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Abstract reachable cells (colored by step 0-{horizon}, '
                 f'x-y projection)')
    ax.legend(loc='upper left')

    sizes = [len(f) for f in frontiers]
    ax2.semilogy(range(len(sizes)), sizes, 'o-')
    ax2.set_xlabel('step')
    ax2.set_ylabel('# abstract states reachable')
    ax2.set_title('Nondeterministic fan-out')
    ax2.grid(True, which='both', alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":

    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Build the unicycle abstraction, summarize it, and "
                    "overlay concrete and abstract rollouts."
    )
    parser.add_argument("--shape", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--rollouts", type=int, default=10)
    parser.add_argument("--x0", type=float, nargs=3, default=[5.0, 10.0, 0.0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show", action="store_true")
    cli_args = parser.parse_args()

    abstraction_shape = [cli_args.shape] * 3
    domain_lb = np.array([0.0, 0.0, -np.pi])
    domain_ub = np.array([50.0, 50.0, np.pi])
    goal_center, goal_radius = np.array([40.0, 20.0]), 8.0
    obs_center, obs_radius = np.array([25.0, 25.0]), 5.0

    # Build the abstraction on a uniform grid
    x_edges, y_edges, theta_edges = [
        np.linspace(lb, ub, n + 1)
        for lb, ub, n in zip(domain_lb, domain_ub, abstraction_shape)
    ]
    kripke_components = build_abstraction(
        x_edges, y_edges, theta_edges,
        verbose=True,
        goal_center=goal_center, goal_radius=goal_radius,
        obs_center=obs_center, obs_radius=obs_radius,
    )

    # Model statistics
    stats = summarize_abstraction(kripke_components)
    print("Model statistics")
    print(f"  States:                 {stats['states']:,}")
    print(f"  Transitions:            {stats['transitions']:,}")
    print(f"  Out-degree (mean/min/max): {stats['out_degree_mean']:.2f} / "
          f"{stats['out_degree_min']} / {stats['out_degree_max']}")
    print(f"  Cell self-loops:        {stats['cell_self_loops']:,}")
    print(f"  Edges into OOB state:   {stats['edges_to_oob']:,}")
    print(f"  Deadlock states:        {stats['deadlock_states']}")
    print(f"  Labels:                 {stats['label_counts']}\n")

    # Concrete trajectory and its abstract cells
    successors, cells = kripke_to_dicts(
        kripke_components, x_edges, y_edges, theta_edges)
    traj = simulate_concrete(cli_args.x0, cli_args.horizon)
    traj_ids = [locate_cell(s, x_edges, y_edges, theta_edges) for s in traj]

    # Abstract reachable sets from the start cell; check soundness
    frontiers = abstract_frontiers(successors, traj_ids[0], cli_args.horizon)
    print("Step | concrete cell | abstract states reachable | contained")
    for t, (cid, fr) in enumerate(zip(traj_ids, frontiers)):
        print(f"  {t:2d} | {cid:13d} | {len(fr):25d} | {cid in fr}")
    assert all(cid in fr for cid, fr in zip(traj_ids, frontiers)), \
        "Concrete trajectory left the abstract reachable set"

    rollouts = sample_abstract_rollouts(
        successors, traj_ids[0], cli_args.horizon,
        cli_args.rollouts, np.random.default_rng(cli_args.seed))

    out_dir = Path(__file__).parent / 'artifacts' / 'demo'
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / 'abstraction_overlay.png'
    plot_overlay(
        traj, frontiers, rollouts, cells, x_edges, y_edges,
        goal_center=goal_center, goal_radius=goal_radius,
        obs_center=obs_center, obs_radius=obs_radius,
        save_path=save_path, show=cli_args.show)
    print(f"\nSaved figure to {save_path}")
