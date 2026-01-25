# Libraries
import numpy as np
import matplotlib.pyplot as plt
import pyModelChecking as pmc
import pyModelChecking.CTL as CTL
from pyModelChecking.CTL import A, E, G, F, Imply
import time


def decode_cell_state_id(state_id, nstates_2, nstates_3):
    """Inverse of cell_state_id(i,j,k) used in make_kripke_from_params."""
    block = nstates_2 * nstates_3
    i = int(state_id // block)
    rem = int(state_id % block)
    j = int(rem // nstates_3)
    k = int(rem % nstates_3)
    return i, j, k

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def theta_min_arc_intervals(thetas, *, eps=1e-12):
    """Return non-wrapping theta interval(s) in [-pi, pi] covering samples.

    Angles live on S^1. Using raw min/max on wrapped angles can incorrectly
    produce near-2*pi spans when samples straddle the -pi/pi cut.

    Returns either:
      - [(lo, hi)] with lo <= hi, or
      - [(-pi, hi), (lo, pi)] when the minimal arc wraps across the cut.
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

def unicycle_dynamics(state, control, control_bound = np.pi/4):

    # Unicycle model parameters
    delta_t = 0.5
    velocity = 5
    pose_x, pose_y, theta = state

    # Apply control bounds
    control = np.clip(control, -control_bound, control_bound) # heading rate of change

    # Update the state
    next_pose_x = pose_x + (delta_t * velocity * np.cos(theta))
    next_pose_y = pose_y + (delta_t * velocity * np.sin(theta))
    next_theta = theta + (delta_t * control)
    next_theta = wrap_to_pi(next_theta) # normalize angle to [-pi, pi]

    return np.array([next_pose_x, next_pose_y, next_theta])

def state_controller(
    state,
    *,
    goal_center,
    obstacle_centers,
    obstacle_radii,
    # gains / shaping
    k_goal=1.0,
    k_rep=8.0,
    alpha=0.6,
    # heading control
    k_theta=2.5,
    omega_max=np.pi/4,
    # numerical smoothing
    eps=1e-6,
):
    """
    Deterministic smooth controller for Dubins/unicycle:
      1) Build desired planar direction v(p) = v_att + v_rep
      2) Convert to desired heading theta_d
      3) Apply smooth saturated turn rate: omega = omega_max * tanh(k_theta * e_theta)
    """

    px, py, theta = state
    p = np.array([px, py], dtype=float)

    # Attractive component (toward goal)
    v_att = k_goal * (goal_center - p)

    # Repulsive component (sum over discs)
    v_rep = np.zeros(2, dtype=float)
    for c, r in zip(obstacle_centers, obstacle_radii):
        diff = p - c
        dist = np.sqrt(diff[0]**2 + diff[1]**2 + eps)  # smoothed distance to center
        clearance = dist - r  # signed clearance (positive outside)

        # Smooth activation: stronger when near obstacle, decays with clearance
        # exp(-alpha * clearance) grows as you get closer (clearance -> 0+)
        w = np.exp(-alpha * clearance)

        # Direction away from obstacle (unit-ish) with additional smoothing in denom
        # Using dist^3 in denom makes it fall off with distance and avoids singularity.
        v_rep += k_rep * w * diff / (dist**3 + eps)

    v = v_att + v_rep

    # If v is (near) zero, define a deterministic fallback
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-9:
        return 0.0

    # Desired heading from vector field
    theta_d = np.arctan2(v[1], v[0])

    # Heading error (wrapped)
    e_theta = wrap_to_pi(theta_d - theta)

    # Smooth saturated turn-rate command
    omega = omega_max * np.tanh(k_theta * e_theta)
    return float(omega)

# Unicylce system with controller in the loop
def cl_unicycle_dynamics(state):

    obs_center = np.array([25.0, 25.0])
    obs_radius = 5.0
    goal_center = np.array([40.0, 20.0])
    goal_radius = 8.0

    control_input = state_controller(
            state,
            goal_center=goal_center,
            obstacle_centers=np.array([obs_center]),
            obstacle_radii=np.array([obs_radius]),
            k_goal=0.3,
            k_rep=300.0,
            alpha=0.1,
            k_theta=2.0,
            omega_max=np.pi/4,
        )
    next_state = unicycle_dynamics(state, control_input)
    return next_state

def simulate_trajectory(x0, *, steps=250, goal_center=None, goal_radius=None):
    """Roll out the closed-loop unicycle from x0. Returns array [T, 3]."""
    state = np.array(x0, dtype=float)
    traj = [state.copy()]
    for _ in range(steps):
        state = cl_unicycle_dynamics(state)
        traj.append(state.copy())

        if (goal_center is not None) and (goal_radius is not None):
            if np.linalg.norm(state[:2] - goal_center) <= goal_radius:
                break
    return np.array(traj)

# Goal state labeling function
def is_goal_state(vertices, center, radius):

    # Return True iff all corners of the x-space cell are within the goal circle
    for v in vertices:
        if np.linalg.norm(v[:2] - center) > radius:
            return False
    return True

# Obstacle state labeling function
def is_obs_state(vertices, center, radius):

    # Return True iff any vertex is within any obstacle circle
    for v in vertices:
        if np.linalg.norm(v[:2] - center) <= radius:
            return True
    return False

# Out-of-bounds state labeling function
def is_oob_state(vertices, x_bounds, y_bounds):

    x_min, x_max = x_bounds
    y_min, y_max = y_bounds

    # Return True iff any vertex is out of bounds
    for v in vertices:
        if (v[0] < x_min) or (v[0] > x_max) or (v[1] < y_min) or (v[1] > y_max):
            return True
    return False

# Helper to find intersecting cells from an AABB in (x, y, theta)-space
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
):
    """Return all grid cell indices (i, j, k) intersected by an axis-aligned box.

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


def make_kripke_from_params(x_params, y_params, theta_params, allow_self_loops=True, advanced_metrics=False):

    # Environment details
    obs_center = np.array([25.0, 25.0])
    obs_radius = 5.0
    goal_center = np.array([40.0, 20.0])
    goal_radius = 8.0

    # Initialize Kripke structure parameters
    nstates_1 = len(x_params) - 1
    nstates_2 = len(y_params) - 1
    nstates_3 = len(theta_params) - 1
    n_kripke_states = nstates_1 * nstates_2 * nstates_3 + 1 # includes out of bounds state
    oob_state_id = n_kripke_states - 1

    # Initialize Kripike structure components
    kripke_states = list(range(n_kripke_states))  # last state is out-of-bounds
    kripke_transitions = set()
    kripke_labels = {}

    # ID labeling function
    def cell_state_id(i, j, k):
        return i * (nstates_2 * nstates_3) + j * nstates_3 + k
    
    # Determine cell volume
    if advanced_metrics:

        # Cell volume using average cell wall lengths
        dx = np.mean(np.diff(x_params))
        dy = np.mean(np.diff(y_params))
        dtheta = np.mean(np.diff(theta_params))
        cell_volume = dx * dy * dtheta

    # Loop through each abstract state
    print("Labeling states and building transitions...")
    count = 0
    succ_count = 0
    self_loop_count = 0
    image_volume = 0.0
    displacement = 0.0
    intersection_over_vol = 0.0
    for i in range(nstates_1):
        x_lo, x_hi = x_params[i], x_params[i+1]
        for j in range(nstates_2):
            y_lo, y_hi = y_params[j], y_params[j+1]
            for k in range(nstates_3):
                theta_lo, theta_hi = theta_params[k], theta_params[k+1]

                verts = np.array([
                    [x_lo, y_lo, theta_lo],
                    [x_lo, y_hi, theta_lo],
                    [x_hi, y_hi, theta_lo],
                    [x_hi, y_lo, theta_lo],
                    [x_lo, y_lo, theta_hi],
                    [x_lo, y_hi, theta_hi],
                    [x_hi, y_hi, theta_hi],
                    [x_hi, y_lo, theta_hi],
                ])

                # Check label of current cell
                if is_goal_state(verts, center=goal_center, radius=goal_radius):
                    label = ['goal']
                elif is_obs_state(verts, center=obs_center, radius=obs_radius):
                    label = ['fail']
                # elif is_oob_state(verts, x_bounds=(x_params[0], x_params[-1]), y_bounds=(y_params[0], y_params[-1])):
                #     label = ['fail']
                else:
                    label = ['safe']

                # Propagate corners through closed-loop dynamics
                next_verts  = np.array([cl_unicycle_dynamics(vert) for vert in verts])

                # Compute theta image interval(s) using minimal circular arc.
                theta_intervals = theta_min_arc_intervals(next_verts[:, 2])

                # Allocate image volume if requested (approximated as AABB of image)
                if advanced_metrics:

                    # Compute AABB volume
                    x_min_img, x_max_img = float(next_verts[:, 0].min()), float(next_verts[:, 0].max())
                    y_min_img, y_max_img = float(next_verts[:, 1].min()), float(next_verts[:, 1].max())
                    theta_span = float(sum(thi - tlo for (tlo, thi) in theta_intervals))
                    img_volume = (x_max_img - x_min_img) * (y_max_img - y_min_img) * theta_span
                    image_volume += img_volume

                    # Compute average push distance of vertices
                    dists = np.linalg.norm(next_verts - verts, axis=1)
                    displacement += np.mean(dists)

                    # Compute the proportion of the parent cell contained in the image AABB
                    contain_x = max(0.0, min(x_hi, x_max_img) - max(x_lo, x_min_img)) / (x_hi - x_lo)
                    contain_y = max(0.0, min(y_hi, y_max_img) - max(y_lo, y_min_img)) / (y_hi - y_lo)
                    theta_overlap = 0.0
                    for (tlo, thi) in theta_intervals:
                        theta_overlap += max(0.0, min(theta_hi, thi) - max(theta_lo, tlo))
                    contain_theta = theta_overlap / (theta_hi - theta_lo)
                    intersection_over_vol += contain_x * contain_y * contain_theta

                # Identify successor cells
                x_min, x_max = float(next_verts[:, 0].min()), float(next_verts[:, 0].max())
                y_min, y_max = float(next_verts[:, 1].min()), float(next_verts[:, 1].max())
                succ_cells_set = set()
                for (theta_min, theta_max) in theta_intervals:
                    succ_cells_set.update(
                        intersecting_cells_from_aabb(
                            x_params,
                            y_params,
                            theta_params,
                            x_min,
                            y_min,
                            theta_min,
                            x_max,
                            y_max,
                            theta_max,
                        )
                    )
                succ_cells = list(succ_cells_set)
                
                # # Indicate if any successor goes out of bounds
                hits_oob = is_oob_state(next_verts, x_bounds=(x_params[0], x_params[-1]), y_bounds=(y_params[0], y_params[-1]))

                # Allocate relations to Kripke structure components
                src = cell_state_id(i, j, k)
                for (ip, jp, kp) in succ_cells:
                    dst = cell_state_id(ip, jp, kp)
                    if not allow_self_loops and dst == src:
                        continue
                    edge = (src, dst)
                    if edge not in kripke_transitions:
                        kripke_transitions.add(edge)
                        succ_count += 1
                        if dst == src:
                            self_loop_count += 1
                if hits_oob:
                    edge = (src, oob_state_id)
                    if edge not in kripke_transitions:
                        kripke_transitions.add(edge)  # transitions oob
                        succ_count += 1

                # Force self loop if no in-grid successors and no OOB transition
                # if not succ_cells:
                #     print("No succ at cell:", (i, j, k))
                #     # kripke_transitions.add((src, src))

                kripke_labels[src] = label
                # print(f"{src}: {label[0]} cell ({i},{j},{k}) -> {len(succ_cells)} in-grid successors")
                if count % 50000 == 0:
                    print(f"    > Processed {count} / {n_kripke_states - 1} states...")
                count += 1

    # # Label the out-of-bounds state; add self-loop
    kripke_labels[oob_state_id] = ['fail']
    kripke_transitions.add((oob_state_id, oob_state_id))

    # Define initial states (in bounds, non-goal states)
    initial_states = [s for s in kripke_states if s != oob_state_id]

    # Output stats
    print(f"Average successors per state: {succ_count / (n_kripke_states - 1):.2f}")
    print(f"Average self-loops per state: {self_loop_count / (n_kripke_states - 1):.4f}")
    print(f"Average cell volume: {cell_volume:.4f}")
    print(f"Average image volume: {image_volume / (n_kripke_states - 1):.4f}")
    print(f"Average image volume / cell volume: {image_volume / ((n_kripke_states - 1) * cell_volume):.2f}")
    print(f"Average centroid displacement: {displacement / (n_kripke_states - 1):.4f}")
    print(f"Average IoV: {intersection_over_vol / (n_kripke_states - 1):.4f}")

    # Make the Kripke structure
    print("Constructing Kripke structure...")
    kripke_structure = pmc.Kripke(S=kripke_states,
                                  S0=initial_states,
                                  R=list(kripke_transitions),
                                  L=kripke_labels)
    
    return kripke_structure


def model_check_kripke(kripke_structure):

    start_cpu = time.process_time()
    phi = 'A (safe U goal)'  # Safe until reach goal
    # phi = 'A (F goal)'  # Safe until reach goal
    sat = CTL.modelcheck(kripke_structure, phi)
    end_cpu = time.process_time()
    cpu_time = end_cpu - start_cpu
    print(f"Model checking CPU time (s): {cpu_time:.2f}")
    
    return sat


def plot_sat_voxels(
    x_params,
    y_params,
    theta_params,
    sat_states,
    *,
    title="Model checking: sat (green) / unsat (red)",
    plot_unsat=True,
    unsat_alpha=0.03,
    sat_alpha=0.85,
    unsat_max_voxels=25000,
    unsat_sample_seed=0,
    edgecolor=None,
):
    """3D voxel plot of the grid cells colored by satisfaction.

    Notes:
    - Plotting *all* unsat cells can be slow for large grids; by default we will
      downsample unsat voxels if there are too many (controlled by unsat_max_voxels).
    - Axes are (x, y, theta) using the provided bin edges.
    """

    x_params = np.asarray(x_params, dtype=float)
    y_params = np.asarray(y_params, dtype=float)
    theta_params = np.asarray(theta_params, dtype=float)

    nx = len(x_params) - 1
    ny = len(y_params) - 1
    ntheta = len(theta_params) - 1
    total = nx * ny * ntheta

    sat_set = set(int(s) for s in sat_states)
    oob_state_id = total  # construction uses +1 oob as last state

    sat_mask = np.zeros((nx, ny, ntheta), dtype=bool)
    for s in sat_set:
        if s == oob_state_id:
            continue
        i, j, k = decode_cell_state_id(s, ny, ntheta)
        if 0 <= i < nx and 0 <= j < ny and 0 <= k < ntheta:
            sat_mask[i, j, k] = True

    if plot_unsat:
        if total <= unsat_max_voxels:
            filled = np.ones((nx, ny, ntheta), dtype=bool)
            colors = np.zeros(filled.shape + (4,), dtype=float)
            colors[..., :] = (1.0, 0.0, 0.0, float(unsat_alpha))
            colors[sat_mask] = (0.0, 0.7, 0.0, float(sat_alpha))
        else:
            # Downsample unsat cells for responsiveness.
            rng = np.random.default_rng(unsat_sample_seed)
            filled = sat_mask.copy()
            colors = np.zeros(filled.shape + (4,), dtype=float)
            colors[sat_mask] = (0.0, 0.7, 0.0, float(sat_alpha))

            unsat_needed = max(0, unsat_max_voxels - int(sat_mask.sum()))
            if unsat_needed > 0:
                unsat_indices = np.argwhere(~sat_mask)
                if len(unsat_indices) > 0:
                    pick = rng.choice(len(unsat_indices), size=min(unsat_needed, len(unsat_indices)), replace=False)
                    picked = unsat_indices[pick]
                    filled[tuple(picked.T)] = True
                    colors[tuple(picked.T)] = (1.0, 0.0, 0.0, float(unsat_alpha))
    else:
        filled = sat_mask
        colors = np.zeros(filled.shape + (4,), dtype=float)
        colors[sat_mask] = (0.0, 0.7, 0.0, float(sat_alpha))

    # Build voxel grid coordinates from bin edges (supports non-uniform grids)
    X, Y, Z = np.meshgrid(x_params, y_params, theta_params, indexing="ij")

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.voxels(X, Y, Z, filled, facecolors=colors, edgecolor=edgecolor)

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("theta")

    ax.set_xlim(x_params[0], x_params[-1])
    ax.set_ylim(y_params[0], y_params[-1])
    ax.set_zlim(theta_params[0], theta_params[-1])

    plt.tight_layout()
    plt.show()




def compute_sat_coverage(sat_ids, x1_params, x2_params, x3_params):

    x1_min = x1_params[0]
    x1_max = x1_params[-1]
    x2_min = x2_params[0]
    x2_max = x2_params[-1]
    x3_min = x3_params[0]
    x3_max = x3_params[-1]
    total_volume = (x1_max - x1_min) * (x2_max - x2_min) * (x3_max - x3_min)

    nstates_1 = len(x1_params) - 1
    nstates_2 = len(x2_params) - 1
    nstates_3 = len(x3_params) - 1

    # ID labeling function
    def cell_state_id(i, j, k):
        return i * (nstates_2 * nstates_3) + j * nstates_3 + k

    # Loop through each abstract state
    sat_volume = 0.0
    for i in range(nstates_1):
        x1_lo, x1_hi = x1_params[i], x1_params[i+1]
        for j in range(nstates_2):
            x2_lo, x2_hi = x2_params[j], x2_params[j+1]
            for k in range(nstates_3):
                x3_lo, x3_hi = x3_params[k], x3_params[k+1]

                if cell_state_id(i, j, k) in sat_ids:
                    sat_volume += (x1_hi - x1_lo) * (x2_hi - x2_lo) * (x3_hi - x3_lo)
    
    return sat_volume / total_volume




def get_gt_reach_regions(domain, grid_resolution=100, verbose=False):

    # Environment details
    obs_center = np.array([25.0, 25.0])
    obs_radius = 5.0
    goal_center = np.array([40.0, 20.0])
    goal_radius = 8.0

    # Unpack domain parameters
    x1_min, x1_max, x2_min, x2_max, x3_min, x3_max = domain

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

                    if is_obs_state(verts, center=obs_center, radius=obs_radius):
                        gt_reach_regions[(i, j, k)] = 'fail'
                        reached_terminal = True
                        break

                    if is_goal_state(verts, goal_center, goal_radius):
                        gt_reach_regions[(i, j, k)] = 'goal'
                        reached_terminal = True
                        break

                    if is_oob_state(verts, x_bounds=(x1_min, x1_max), y_bounds=(x2_min, x2_max)):
                        gt_reach_regions[(i, j, k)] = 'fail'
                        reached_terminal = True
                        break

                    verts = np.array([cl_unicycle_dynamics(vert) for vert in verts])

                if not reached_terminal:
                    gt_reach_regions[(i, j, k)] = 'unk'
                
                if verbose and (count % 10000 == 0):
                    print(f"    > Processed {count} / {total_cells} regions...")
                count += 1
    
    return gt_reach_regions




def check_ground_truth_fast(x1_params, x2_params, x3_params, domain, gt_reach_regions):

    """Classify abstraction cells against a fixed labeled 3D grid.

    Args:
        x1_params, x2_params, x3_params: bin edges for the abstraction grid.
        domain: (x1_min, x1_max, x2_min, x2_max, x3_min, x3_max)
        gt_reach_regions: dict mapping (i,j,k) fixed-grid cell indices to
            labels in {'fail','goal','unk'}.

    Labeling rule implemented (per your spec):
      - 'goal' iff the abstraction cell is fully contained in the union of
        fixed-grid cells labeled 'goal'.
      - otherwise 'fail' (we don't try to be clever about returning 'unk').
    """

    if not gt_reach_regions:
        raise ValueError("gt_reach_regions is empty")

    # Unpack domain parameters
    x1_min, x1_max, x2_min, x2_max, x3_min, x3_max = map(float, domain)

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

    nstates_1 = len(x1_params) - 1
    nstates_2 = len(x2_params) - 1
    nstates_3 = len(x3_params) - 1

    def cell_state_id(i, j, k):
        return i * (nstates_2 * nstates_3) + j * nstates_3 + k

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
        a1_lo, a1_hi = float(x1_params[i]), float(x1_params[i + 1])
        for j in range(nstates_2):
            a2_lo, a2_hi = float(x2_params[j]), float(x2_params[j + 1])
            for k in range(nstates_3):
                a3_lo, a3_hi = float(x3_params[k]), float(x3_params[k + 1])

                src = cell_state_id(i, j, k)

                # If abstraction cell is out of the domain, mark fail.
                if (
                    (a1_lo < x1_min)
                    or (a1_hi > x1_max)
                    or (a2_lo < x2_min)
                    or (a2_hi > x2_max)
                    or (a3_lo < x3_min)
                    or (a3_hi > x3_max)
                ):
                    ground_truth_check[src] = "fail"
                    continue

                r1 = _fixed_index_range(x1_vals, a1_lo, a1_hi)
                r2 = _fixed_index_range(x2_vals, a2_lo, a2_hi)
                r3 = _fixed_index_range(x3_vals, a3_lo, a3_hi)
                if r1 is None or r2 is None or r3 is None:
                    ground_truth_check[src] = "fail"
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

                ground_truth_check[src] = "goal" if all_goal else "fail"

    return ground_truth_check




def compute_sat_coverage(sat_ids, x1_params, x2_params, x3_params):

    x1_min = x1_params[0]
    x1_max = x1_params[-1]
    x2_min = x2_params[0]
    x2_max = x2_params[-1]
    x3_min = x3_params[0]
    x3_max = x3_params[-1]
    total_area = (x1_max - x1_min) * (x2_max - x2_min) * (x3_max - x3_min)

    nstates_1 = len(x1_params) - 1
    nstates_2 = len(x2_params) - 1
    nstates_3 = len(x3_params) - 1

    # ID labeling function
    def cell_state_id(i, j, k):
        return i * (nstates_2 * nstates_3) + j * nstates_3 + k

    # Loop through each abstract state
    sat_area = 0.0
    for i in range(nstates_1):
        x1_lo, x1_hi = x1_params[i], x1_params[i+1]
        for j in range(nstates_2):
            x2_lo, x2_hi = x2_params[j], x2_params[j+1]
            for k in range(nstates_3):
                x3_lo, x3_hi = x3_params[k], x3_params[k+1]

                if cell_state_id(i, j, k) in sat_ids:
                    sat_area += (x1_hi - x1_lo) * (x2_hi - x2_lo) * (x3_hi - x3_lo)
    
    return sat_area / total_area


def false_negative_rate(true_safe_states, checked_safe_states):
    false_negative_states = {s for s in true_safe_states if s not in checked_safe_states}
    denom = len(true_safe_states)
    fnr = (len(false_negative_states) / denom) if denom > 0 else float('nan')
    return fnr, false_negative_states


