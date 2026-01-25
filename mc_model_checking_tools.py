# Libraries
import numpy as np
import pyModelChecking as pmc
import grid_plot_tools as gpt
import pyModelChecking.CTL as CTL
from pyModelChecking.CTL import A, E, G, F, Imply
import torch
import torch.nn as nn
from pathlib import Path
import time


# MountainCar open-loop dynamics
def mc_ol_dynamics(state, action):
    p, v = state
    v_next = v + 5*0.001*(action - 1) - 0.0025*np.cos(3*p) # differential is scaled for coarser simulation
    v_next = np.clip(v_next, -0.07, 0.07)
    p_next = p + v_next
    p_next = np.clip(p_next, -1.2, 0.6)
    if p_next <= -1.2 and v_next < 0:
        v_next = 0.0
    return np.array([p_next, v_next])

# DQN policy
class DQN(nn.Module):

    # Initialize the neural network
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), # Input layer
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), # Hidden layer
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim) # Output layer
        )

    def forward(self, x):
        return self.net(x)

# Lazy-load cached policy (avoid re-instantiation)
_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_POLICY_CACHE = None
_POLICY_PATH = Path(__file__).resolve().parent / 'policy.pth'

def get_policy():
    global _POLICY_CACHE
    if _POLICY_CACHE is None:
        net = DQN(2, 3, 128).to(_DEVICE)
        state_dict = torch.load(str(_POLICY_PATH), map_location=_DEVICE)
        net.load_state_dict(state_dict)
        net.eval()
        _POLICY_CACHE = net
    return _POLICY_CACHE

def policy_action(state):
    net = get_policy()
    with torch.no_grad():
        s = torch.as_tensor(state, dtype=torch.float32, device=_DEVICE).unsqueeze(0)
        q = net(s)
        return int(q.argmax(dim=1).item())

# Closed-loop dynamics using greedy policy (no need to pass network each call)
def mc_cl_dynamics(x):
    x_np = np.asarray(x, dtype=np.float64)
    a_idx = policy_action(x_np)
    return mc_ol_dynamics(x_np, a_idx)


# Helper to find intersecting cells from AABB in y-space
def intersecting_cells_from_y_aabb(y1_params, y2_params, y1_min, y1_max, y2_min, y2_max):
    """Sound over-approx: return all (i,j) whose Y-cell rectangle overlaps the given AABB.
    """
    y1_params = np.asarray(y1_params, dtype=float)
    y2_params = np.asarray(y2_params, dtype=float)
    n1 = len(y1_params) - 1
    n2 = len(y2_params) - 1

    # If completely outside the overall grid bounds, no in-grid successors.
    grid_y1_min, grid_y1_max = float(y1_params[0]), float(y1_params[-1])
    grid_y2_min, grid_y2_max = float(y2_params[0]), float(y2_params[-1])
    if y1_max < grid_y1_min or y1_min > grid_y1_max or y2_max < grid_y2_min or y2_min > grid_y2_max:
        return []

    # Use side="right" for both ends.
    # This avoids the empty-set edge case when y_min == y_max equals a grid line:
    #   side="left" would assign the max to the lower cell, potentially giving i_hi < i_lo.
    i_lo = int(np.searchsorted(y1_params, y1_min, side="right") - 1)
    i_hi = int(np.searchsorted(y1_params, y1_max, side="right") - 1)
    j_lo = int(np.searchsorted(y2_params, y2_min, side="right") - 1)
    j_hi = int(np.searchsorted(y2_params, y2_max, side="right") - 1)

    i_lo = max(0, min(n1 - 1, i_lo))
    i_hi = max(0, min(n1 - 1, i_hi))
    j_lo = max(0, min(n2 - 1, j_lo))
    j_hi = max(0, min(n2 - 1, j_hi))

    if i_hi < i_lo or j_hi < j_lo:
        return []

    return [(i, j) for i in range(i_lo, i_hi + 1) for j in range(j_lo, j_hi + 1)]


# Helper to check if point is in valid y domain
def point_in_parallelogram(p, verts, tol=1e-9):
    """Return True iff point p is inside/on the parallelogram defined by verts (CCW order)."""
    verts = np.asarray(verts, dtype=float)
    p = np.asarray(p, dtype=float)

    v0 = verts[0]
    e1 = verts[1] - v0
    e2 = verts[-1] - v0
    E = np.column_stack([e1, e2])  # 2x2

    # Solve p = v0 + a*e1 + b*e2
    ab = np.linalg.solve(E, p - v0)
    a, b = float(ab[0]), float(ab[1])
    return (-tol <= a <= 1.0 + tol) and (-tol <= b <= 1.0 + tol)


# Goal state labeling function
def is_goal_state(x_space_vertices):
    for v in x_space_vertices:
        if v[0] < 0.5:
            return False
    return True


def make_kripke_from_params(y1_params, y2_params, M):

    # Initialize Kripke structure parameters
    nstates_1 = len(y1_params) - 1
    nstates_2 = len(y2_params) - 1
    n_kripke_states = nstates_1 * nstates_2

    # Initialize Kripike structure components
    kripke_states = list(range(n_kripke_states))
    kripke_transitions = set()
    kripke_labels = {}

    # ID labeling function
    def cell_state_id(i, j):
        return i * nstates_2 + j

    # Precompute transformation inverse
    M = np.asarray(M, dtype=float)
    invM = np.linalg.inv(M)

    # Loop through each abstract state
    count = 0
    for i in range(nstates_1):
        y1_lo, y1_hi = y1_params[i], y1_params[i+1]
        for j in range(nstates_2):
            y2_lo, y2_hi = y2_params[j], y2_params[j+1]
            corners = np.array([
                [y1_lo, y2_lo],
                [y1_lo, y2_hi],
                [y1_hi, y2_hi],
                [y1_hi, y2_lo]])

            # Compute y-space image
            x_corners = (M @ corners.T).T
            x_next = np.array([mc_cl_dynamics(x_corner) for x_corner in x_corners])
            y_next = (invM @ x_next.T).T

            # Check label of current cell
            label = ['goal'] if is_goal_state(x_corners) else ['safe']

            # Identify successor cells
            y1_min, y1_max = float(y_next[:, 0].min()), float(y_next[:, 0].max())
            y2_min, y2_max = float(y_next[:, 1].min()), float(y_next[:, 1].max())
            succ_cells = intersecting_cells_from_y_aabb(y1_params, y2_params, y1_min, y1_max, y2_min, y2_max)

            # Allocate relations to Kripke structure components
            src = cell_state_id(i, j)
            for (ip, jp) in succ_cells:
                dst = cell_state_id(ip, jp)
                kripke_transitions.add((src, dst))

            # # Force self loop if no in-grid successors
            # if (not succ_cells):
            #     kripke_transitions.add((src, src))

            kripke_labels[src] = label

            # if count % 1000 == 0:
                # print(f"Processed {count} / {n_kripke_states} Kripke states...")
            # print(f"{src}: {label[0]} cell ({i},{j}) -> {len(succ_cells)} in-grid successors" + (" + OOB" if hits_oob else ""))
            count += 1

    # Define initial states (in bounds, non-goal states)
    initial_states = [s for s in kripke_states]

    # Make the Kripke structure
    kripke_structure = pmc.Kripke(S=kripke_states,
                                  S0=initial_states,
                                  R=list(kripke_transitions),
                                  L=kripke_labels)
    
    return kripke_structure





def make_kripke(x1_params, x2_params, allow_self_loops=True, advanced_metrics=False, verbose=False):

    # Initialize Kripke structure parameters
    nstates_1 = len(x1_params) - 1
    nstates_2 = len(x2_params) - 1
    n_kripke_states = nstates_1 * nstates_2

    # Initialize Kripike structure components
    kripke_states = list(range(n_kripke_states))
    kripke_transitions = set()
    kripke_labels = {}

    # ID labeling function
    def cell_state_id(i, j):
        return i * nstates_2 + j

    # Determine cell area
    if advanced_metrics:

        # Cell area using average cell wall lengths
        dy1 = np.mean(np.diff(x1_params))
        dy2 = np.mean(np.diff(x2_params))
        cell_area = dy1 * dy2

    # Loop through each abstract state
    if verbose:
        print("Labeling states and building transitions...")
    count = 0
    succ_count = 0
    self_loop_count = 0
    image_area = 0.0
    displacement = 0.0
    intersection_over_area = 0.0
    intersection_over_union = 0.0
    for i in range(nstates_1):
        x1_lo, x1_hi = x1_params[i], x1_params[i+1]
        for j in range(nstates_2):
            x2_lo, x2_hi = x2_params[j], x2_params[j+1]
            corners = np.array([
                [x1_lo, x2_lo],
                [x1_lo, x2_hi],
                [x1_hi, x2_hi],
                [x1_hi, x2_lo]])

            # Compute y-space image
            x_next =  np.array([mc_cl_dynamics(corner) for corner in corners])

            # Check label of current cell
            label = ['goal'] if is_goal_state(corners) else ['safe']

            # Allocate image area if requested (approximated as AABB of image)
            if advanced_metrics:

                # Compute AABB side lengths of parent and image
                x1_lo_pre, x1_hi_pre = float(corners[:, 0].min()), float(corners[:, 0].max())
                x2_lo_pre, x2_hi_pre = float(corners[:, 1].min()), float(corners[:, 1].max())
                x1_lo_post, x1_hi_post = float(x_next[:, 0].min()), float(x_next[:, 0].max())
                x2_lo_post, x2_hi_post = float(x_next[:, 1].min()), float(x_next[:, 1].max())

                # Compute image AABB area
                img_area = (x1_hi_post - x1_lo_post) * (x2_hi_post - x2_lo_post)
                image_area += img_area

                # Compute parent AABB area
                parent_area = (x1_hi_pre - x1_lo_pre) * (x2_hi_pre - x2_lo_pre)

                # Compute displacement between centroids of parent and image AABB
                dists = np.linalg.norm(x_next - corners, axis=1)
                displacement += np.mean(dists)

                # Compute the proportion of the parent cell contained in the image AABB
                l1 = max(0, min(x1_hi_pre, x1_hi_post) - max(x1_lo_pre, x1_lo_post))
                l2 = max(0, min(x2_hi_pre, x2_hi_post) - max(x2_lo_pre, x2_lo_post))
                int_area = (l1 * l2)
                intersection_over_area += int_area / parent_area

                # Compute the proportion of intersection area to union area
                union_area = parent_area + image_area - int_area
                intersection_over_union += int_area / union_area
            
            # Identify successor cells
            x1_min, x1_max = float(x_next[:, 0].min()), float(x_next[:, 0].max())
            x2_min, x2_max = float(x_next[:, 1].min()), float(x_next[:, 1].max())
            succ_cells = intersecting_cells_from_y_aabb(x1_params, x2_params, x1_min, x1_max, x2_min, x2_max)

            # Allocate relations to Kripke structure components
            src = cell_state_id(i, j)
            for (ip, jp) in succ_cells:
                dst = cell_state_id(ip, jp)
                if not allow_self_loops and dst == src and len(succ_cells) > 1:
                        continue
                edge = (src, dst)
                if edge not in kripke_transitions:
                    kripke_transitions.add(edge)
                    succ_count += 1
                    if dst == src:
                        self_loop_count += 1

            # # Force self loop if no in-grid successors and no OOB transition
            # if not succ_cells:
            #     kripke_transitions.add((src, src))

            kripke_labels[src] = label
            if verbose and count % 10000 == 0:
                    print(f"    > Processed {count} / {n_kripke_states - 1} states...")
            count += 1

    # Define initial states (in bounds, non-goal states)
    initial_states = [s for s in kripke_states]

    # Output stats
    if verbose:
        print(f"Average successors per state: {succ_count / (n_kripke_states - 1):.2f}")
        print(f"Average self-loops per state: {self_loop_count / (n_kripke_states - 1):.4f}")
        print(f"Average cell area: {cell_area:.4f}")
        print(f"Average image area: {image_area / (n_kripke_states - 1):.4f}")
        print(f"Average image area / cell area: {image_area / ((n_kripke_states - 1) * cell_area):.2f}")
        print(f"Average centroid displacement: {displacement / (n_kripke_states - 1):.4f}")
        print(f"Average IoA: {intersection_over_area / (n_kripke_states - 1):.4f}")
        print(f"Average IoU: {intersection_over_union / (n_kripke_states - 1):.4f}")

    # Make the Kripke structure
    if verbose:
        print("Constructing Kripke structure...")
    kripke_structure = pmc.Kripke(S=kripke_states,
                                  S0=initial_states,
                                  R=list(kripke_transitions),
                                  L=kripke_labels)
    
    return kripke_structure





def model_check_kripke(kripke_structure):

    start_cpu = time.process_time()
    # print("Starting model checking...")
    phi = 'A (F goal)'  # Eventually reach goal
    sat = CTL.modelcheck(kripke_structure, phi)
    end_cpu = time.process_time()
    cpu_time = end_cpu - start_cpu
    print(f"Model checking CPU time (s): {cpu_time:.2f}")
    
    return sat



def fixed_check_ground_truth(x_domain, y1_params, y2_params, M, grid_resolution=100):

    # Unpack domain parameters
    x1_min, x1_max, x2_min, x2_max = map(float, x_domain)

    # Initialize a uniform grid over the x-domain
    x1_vals = np.linspace(x1_min, x1_max, grid_resolution)
    x2_vals = np.linspace(x2_min, x2_max, grid_resolution)

    # Iterate through each grid cell; classify as 'goal' or 'unk'
    fixed_check = {}
    for i in range(grid_resolution - 1):
        for j in range(grid_resolution - 1):
            x1_lo, x1_hi = x1_vals[i], x1_vals[i+1]
            x2_lo, x2_hi = x2_vals[j], x2_vals[j+1]

            # Define cell corners
            corners = np.array([
                [x1_lo, x2_lo],
                [x1_lo, x2_hi],
                [x1_hi, x2_hi],
                [x1_hi, x2_lo]])
            
            # Push vertices through dynamics until any fail or all succeed
            max_steps = 10_000
            reached_terminal = False
            for _ in range(max_steps):

                if is_goal_state(corners):
                    fixed_check[(i, j)] = 'goal'
                    reached_terminal = True
                    # print(f"Fixed cell ({i},{j}): all goal reached")
                    break

                corners = np.array([mc_cl_dynamics(v) for v in corners])

            if not reached_terminal:
                fixed_check[(i, j)] = 'unk'
                # print(f"Fixed cell ({i},{j}): unknown")

    # --- Now classify the user's affine (y-grid) cells using the fixed x-grid labels ---
    nstates_1 = len(y1_params) - 1
    nstates_2 = len(y2_params) - 1

    # ID labeling function for user abstraction cells
    def cell_state_id(i, j):
        return i * nstates_2 + j

    M = np.asarray(M, dtype=float)

    # Helper: over-approx candidate x-grid cells from an x-space AABB
    def _intersecting_fixed_cells_from_x_aabb(x1_min_aabb, x1_max_aabb, x2_min_aabb, x2_max_aabb):
        n1 = grid_resolution - 1
        n2 = grid_resolution - 1

        # If completely outside x-domain, return empty and let caller decide label.
        if x1_max_aabb < x1_min or x1_min_aabb > x1_max or x2_max_aabb < x2_min or x2_min_aabb > x2_max:
            return []

        # Map AABB limits to grid-cell indices.
        i_lo = int(np.searchsorted(x1_vals, x1_min_aabb, side="right") - 1)
        i_hi = int(np.searchsorted(x1_vals, x1_max_aabb, side="right") - 1)
        j_lo = int(np.searchsorted(x2_vals, x2_min_aabb, side="right") - 1)
        j_hi = int(np.searchsorted(x2_vals, x2_max_aabb, side="right") - 1)

        i_lo = max(0, min(n1 - 1, i_lo))
        i_hi = max(0, min(n1 - 1, i_hi))
        j_lo = max(0, min(n2 - 1, j_lo))
        j_hi = max(0, min(n2 - 1, j_hi))

        if i_hi < i_lo or j_hi < j_lo:
            return []

        return [(ii, jj) for ii in range(i_lo, i_hi + 1) for jj in range(j_lo, j_hi + 1)]

    # Helper: convex polygon intersection via Separating Axis Theorem (SAT)
    def _poly_intersects_poly(poly_a, poly_b, tol=1e-12):
        poly_a = np.asarray(poly_a, dtype=float)
        poly_b = np.asarray(poly_b, dtype=float)

        def _edges(poly):
            return poly[(np.arange(len(poly)) + 1) % len(poly)] - poly

        def _axes_from_edges(edges):
            # Perpendicular normals for each edge
            return np.column_stack([-edges[:, 1], edges[:, 0]])

        def _project(poly, axis):
            axis = np.asarray(axis, dtype=float)
            norm = np.linalg.norm(axis)
            if norm < tol:
                return None
            axis = axis / norm
            dots = poly @ axis
            return float(dots.min()), float(dots.max())

        axes = np.vstack([
            _axes_from_edges(_edges(poly_a)),
            _axes_from_edges(_edges(poly_b)),
        ])

        for axis in axes:
            pa = _project(poly_a, axis)
            pb = _project(poly_b, axis)
            if pa is None or pb is None:
                continue
            a_min, a_max = pa
            b_min, b_max = pb
            if a_max < b_min - tol or b_max < a_min - tol:
                return False
        return True

    # Helper: parallelogram (x-space) intersects x-rectangle cell (axis-aligned)
    def _parallelogram_intersects_cell(par_verts, cell_i, cell_j):
        x1_lo, x1_hi = x1_vals[cell_i], x1_vals[cell_i + 1]
        x2_lo, x2_hi = x2_vals[cell_j], x2_vals[cell_j + 1]
        rect = np.array([
            [x1_lo, x2_lo],
            [x1_lo, x2_hi],
            [x1_hi, x2_hi],
            [x1_hi, x2_lo],
        ], dtype=float)
        return _poly_intersects_poly(par_verts, rect)

    ground_truth_check = {}

    for i in range(nstates_1):
        y1_lo, y1_hi = y1_params[i], y1_params[i + 1]
        for j in range(nstates_2):
            y2_lo, y2_hi = y2_params[j], y2_params[j + 1]

            # y-cell corners (axis-aligned in y-space)
            y_corners = np.array([
                [y1_lo, y2_lo],
                [y1_lo, y2_hi],
                [y1_hi, y2_hi],
                [y1_hi, y2_lo],
            ], dtype=float)

            # Map to x-space: x = M y (affine cell becomes a parallelogram in x)
            x_par = (M @ y_corners.T).T

            src = cell_state_id(i, j)

            # # If any vertex is outside the x-domain, conservatively mark as fail
            # if np.any(x_par[:, 0] < x1_min) or np.any(x_par[:, 0] > x1_max) or np.any(x_par[:, 1] < x2_min) or np.any(x_par[:, 1] > x2_max):
            #     ground_truth_check[src] = 'fail'
            #     continue

            # Candidate fixed cells from AABB over-approx
            x1_min_aabb, x1_max_aabb = float(x_par[:, 0].min()), float(x_par[:, 0].max())
            x2_min_aabb, x2_max_aabb = float(x_par[:, 1].min()), float(x_par[:, 1].max())
            candidates = _intersecting_fixed_cells_from_x_aabb(x1_min_aabb, x1_max_aabb, x2_min_aabb, x2_max_aabb)

            if not candidates:
                ground_truth_check[src] = 'unk'
                continue

            # Check actual intersections and aggregate labels
            seen_fail = False
            seen_unk = False
            seen_goal = False

            for (ii, jj) in candidates:
                if not _parallelogram_intersects_cell(x_par, ii, jj):
                    continue
                lab = fixed_check.get((ii, jj), 'unk')
                if lab == 'fail':
                    seen_fail = True
                    break
                if lab == 'unk':
                    seen_unk = True
                elif lab == 'goal':
                    seen_goal = True

            if seen_fail:
                ground_truth_check[src] = 'fail'
                print(f"Cell ({i},{j}): some fail reached")
            elif seen_unk:
                ground_truth_check[src] = 'unk'
                print(f"Cell ({i},{j}): some unk reached")
            else:
                # If it intersects only goal-labeled fixed cells, declare goal. If it
                # didn't intersect anything due to numerical edge cases, keep unk.
                ground_truth_check[src] = 'goal' if seen_goal else 'unk'
                # print(f"Cell ({i},{j}): all goal reached")

    return ground_truth_check



def check_ground_truth(y1_params, y2_params, M):

    # Initialize ground truth dictionary
    nstates_1 = len(y1_params) - 1
    nstates_2 = len(y2_params) - 1
    ground_truth_check = {}

    # ID labeling function
    def cell_state_id(i, j):
        return i * nstates_2 + j

    # Precompute transformation matrix
    M = np.asarray(M, dtype=float)

    # Loop through each abstract state
    for i in range(nstates_1):
        y1_lo, y1_hi = y1_params[i], y1_params[i+1]
        for j in range(nstates_2):
            y2_lo, y2_hi = y2_params[j], y2_params[j+1]

            # Transform corners to x-space
            corners = np.array([
                [y1_lo, y2_lo],
                [y1_lo, y2_hi],
                [y1_hi, y2_hi],
                [y1_hi, y2_lo]])
            x_verts = (M @ corners.T).T

            # Push vertices through dynamics until any fail or all succeed
            src = cell_state_id(i, j)
            max_steps = 10_000
            reached_terminal = False
            for _ in range(max_steps):

                if is_goal_state(x_verts):
                    # print(f"Cell ({i},{j}): all goal reached")
                    ground_truth_check[src] = 'goal'
                    reached_terminal = True
                    break

                x_verts = np.array([mc_cl_dynamics(v) for v in x_verts])

            if not reached_terminal:
                ground_truth_check[src] = 'unk'

            # print(f"Cell ({i},{j}): " +
            #       ("OOB reached" if any_oob else
            #        "All goal reached" if all_goal else
            #        "Neither OOB nor all goal reached"))

    return ground_truth_check

if __name__ == "__main__":

    K = pmc.Kripke(S=[0, 1, 3],
                R=[(0, 2), (2, 2), (0, 1), (1, 0), (3, 2)],
                L={1: ['p', 'q'], 2: ['p', 'q'], 3: ['q']})

    print(K.states())
    print(K.transitions())

def false_negative_rate(true_safe_states, checked_safe_states):
    false_negative_states = {s for s in true_safe_states if s not in checked_safe_states}
    denom = len(true_safe_states)
    fnr = (len(false_negative_states) / denom) if denom > 0 else float('nan')
    return fnr, false_negative_states






def get_gt_reach_regions(x_domain, grid_resolution):

    # Unpack domain parameters
    x1_min, x1_max, x2_min, x2_max = map(float, x_domain)

    # Initialize a uniform grid over the x-domain
    x1_vals = np.linspace(x1_min, x1_max, grid_resolution)
    x2_vals = np.linspace(x2_min, x2_max, grid_resolution)

    # Iterate through each grid cell; classify as 'fail', 'goal', or 'unk'
    gt_reach_regions = {}
    for i in range(grid_resolution - 1):
        for j in range(grid_resolution - 1):
            x1_lo, x1_hi = x1_vals[i], x1_vals[i+1]
            x2_lo, x2_hi = x2_vals[j], x2_vals[j+1]

            # Define cell corners
            corners = np.array([
                [x1_lo, x2_lo],
                [x1_lo, x2_hi],
                [x1_hi, x2_hi],
                [x1_hi, x2_lo]])
            
            # Push vertices through dynamics until any fail or all succeed
            max_steps = 10_000
            reached_terminal = False
            for _ in range(max_steps):

                if is_goal_state(corners):
                    gt_reach_regions[(i, j)] = 'goal'
                    reached_terminal = True
                    break

                corners = np.array([mc_cl_dynamics(v) for v in corners])

            if not reached_terminal:
                gt_reach_regions[(i, j)] = 'unk'
    
    return gt_reach_regions


def check_ground_truth_fast(x1_params, x2_params, x_domain, gt_reach_regions):

    # Unpack domain parameters
    x1_min, x1_max, x2_min, x2_max = map(float, x_domain)

    # Initialize a uniform grid over the x-domain
    grid_resolution = int(np.sqrt(len(gt_reach_regions))) + 1
    x1_vals = np.linspace(x1_min, x1_max, grid_resolution)
    x2_vals = np.linspace(x2_min, x2_max, grid_resolution)

    # --- Now classify the user's affine (y-grid) cells using the fixed x-grid labels ---
    nstates_1 = len(x1_params) - 1
    nstates_2 = len(x2_params) - 1

    # ID labeling function for user abstraction cells
    def cell_state_id(i, j):
        return i * nstates_2 + j

    # Helper: over-approx candidate x-grid cells from an x-space AABB
    def _intersecting_fixed_cells_from_x_aabb(x1_min_aabb, x1_max_aabb, x2_min_aabb, x2_max_aabb):
        n1 = grid_resolution - 1
        n2 = grid_resolution - 1

        # If completely outside x-domain, return empty and let caller decide label.
        if x1_max_aabb < x1_min or x1_min_aabb > x1_max or x2_max_aabb < x2_min or x2_min_aabb > x2_max:
            return []

        # Map AABB limits to grid-cell indices.
        i_lo = int(np.searchsorted(x1_vals, x1_min_aabb, side="right") - 1)
        i_hi = int(np.searchsorted(x1_vals, x1_max_aabb, side="right") - 1)
        j_lo = int(np.searchsorted(x2_vals, x2_min_aabb, side="right") - 1)
        j_hi = int(np.searchsorted(x2_vals, x2_max_aabb, side="right") - 1)

        i_lo = max(0, min(n1 - 1, i_lo))
        i_hi = max(0, min(n1 - 1, i_hi))
        j_lo = max(0, min(n2 - 1, j_lo))
        j_hi = max(0, min(n2 - 1, j_hi))

        if i_hi < i_lo or j_hi < j_lo:
            return []

        return [(ii, jj) for ii in range(i_lo, i_hi + 1) for jj in range(j_lo, j_hi + 1)]

    # Helper: convex polygon intersection via Separating Axis Theorem (SAT)
    def _poly_intersects_poly(poly_a, poly_b, tol=1e-12):
        poly_a = np.asarray(poly_a, dtype=float)
        poly_b = np.asarray(poly_b, dtype=float)

        def _edges(poly):
            return poly[(np.arange(len(poly)) + 1) % len(poly)] - poly

        def _axes_from_edges(edges):
            # Perpendicular normals for each edge
            return np.column_stack([-edges[:, 1], edges[:, 0]])

        def _project(poly, axis):
            axis = np.asarray(axis, dtype=float)
            norm = np.linalg.norm(axis)
            if norm < tol:
                return None
            axis = axis / norm
            dots = poly @ axis
            return float(dots.min()), float(dots.max())

        axes = np.vstack([
            _axes_from_edges(_edges(poly_a)),
            _axes_from_edges(_edges(poly_b)),
        ])

        for axis in axes:
            pa = _project(poly_a, axis)
            pb = _project(poly_b, axis)
            if pa is None or pb is None:
                continue
            a_min, a_max = pa
            b_min, b_max = pb
            if a_max < b_min - tol or b_max < a_min - tol:
                return False
        return True

    # Helper: parallelogram (x-space) intersects x-rectangle cell (axis-aligned)
    def _parallelogram_intersects_cell(par_verts, cell_i, cell_j):
        x1_lo, x1_hi = x1_vals[cell_i], x1_vals[cell_i + 1]
        x2_lo, x2_hi = x2_vals[cell_j], x2_vals[cell_j + 1]
        rect = np.array([
            [x1_lo, x2_lo],
            [x1_lo, x2_hi],
            [x1_hi, x2_hi],
            [x1_hi, x2_lo],
        ], dtype=float)
        return _poly_intersects_poly(par_verts, rect)

    ground_truth_check = {}

    for i in range(nstates_1):
        x1_lo, x1_hi = x1_params[i], x1_params[i + 1]
        for j in range(nstates_2):
            x2_lo, x2_hi = x2_params[j], x2_params[j + 1]

            # y-cell corners (axis-aligned in y-space)
            x_par = np.array([
                [x1_lo, x2_lo],
                [x1_lo, x2_hi],
                [x1_hi, x2_hi],
                [x1_hi, x2_lo],
            ], dtype=float)

            src = cell_state_id(i, j)

            # If any vertex is outside the x-domain, conservatively mark as fail
            if np.any(x_par[:, 0] < x1_min) or np.any(x_par[:, 0] > x1_max) or np.any(x_par[:, 1] < x2_min) or np.any(x_par[:, 1] > x2_max):
                ground_truth_check[src] = 'fail'
                continue

            # Candidate fixed cells from AABB over-approx
            x1_min_aabb, x1_max_aabb = float(x_par[:, 0].min()), float(x_par[:, 0].max())
            x2_min_aabb, x2_max_aabb = float(x_par[:, 1].min()), float(x_par[:, 1].max())
            candidates = _intersecting_fixed_cells_from_x_aabb(x1_min_aabb, x1_max_aabb, x2_min_aabb, x2_max_aabb)

            if not candidates:
                ground_truth_check[src] = 'unk'
                continue

            # Check actual intersections and aggregate labels
            seen_fail = False
            seen_unk = False
            seen_goal = False

            for (ii, jj) in candidates:
                if not _parallelogram_intersects_cell(x_par, ii, jj):
                    continue
                lab = gt_reach_regions.get((ii, jj), 'unk')
                if lab == 'fail':
                    seen_fail = True
                    break
                if lab == 'unk':
                    seen_unk = True
                elif lab == 'goal':
                    seen_goal = True

            if seen_fail:
                ground_truth_check[src] = 'fail'
            elif seen_unk:
                ground_truth_check[src] = 'unk'
            else:
                # If it intersects only goal-labeled fixed cells, declare goal. If it
                # didn't intersect anything due to numerical edge cases, keep unk.
                ground_truth_check[src] = 'goal' if seen_goal else 'unk'

    return ground_truth_check


def compute_sat_coverage(sat_ids, x1_params, x2_params):

    x1_min = x1_params[0]
    x1_max = x1_params[-1]
    x2_min = x2_params[0]
    x2_max = x2_params[-1]
    total_area = (x1_max - x1_min) * (x2_max - x2_min)

    nstates_1 = len(x1_params) - 1
    nstates_2 = len(x2_params) - 1

    # ID labeling function
    def cell_state_id(i, j):
        return i * nstates_2 + j

    # Loop through each abstract state
    sat_area = 0.0
    for i in range(nstates_1):
        x1_lo, x1_hi = x1_params[i], x1_params[i+1]
        for j in range(nstates_2):
            x2_lo, x2_hi = x2_params[j], x2_params[j+1]

            if cell_state_id(i, j) in sat_ids:
                sat_area += (x1_hi - x1_lo) * (x2_hi - x2_lo)
    
    return sat_area / total_area